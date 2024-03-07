from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from lvmc.data_handling.data_exporter import DataExporter
from tqdm import tqdm
from utils import *
from rich import print
import numpy as np

# Parameters for ParticleLattice

width = 50
height = 25
g = 1.5
v0 = 100.0
density = 0.3
flow_params = {"type": "poiseuille", "v1": 15}
tmax = 1
    
def main():
    # Initialize the Simulation
    simulation = Simulation(g, v0, width=width, height=height, density=density, flow_params=flow_params, with_transport=False)
    data_collector = DataCollector(simulation)
    fname_dumps = "test.hdf5"
    fname_stats = "test.txt"
    data_exporter = DataExporter(fname_dumps, data_collector)


    obstacles = torch.zeros((height, width), dtype=torch.bool)
    obstacles[0, :] = True
    obstacles[-1, :] = True
    simulation.lattice.set_obstacles(obstacles)

    dt_flow = 0.1/flow_params["v1"]
    dt_dump = 0.1
    tlast_flow = 0
    tlast_dump = 0
    simulation.init_stat()
    cnt = 0

    while simulation.t < tmax:
        event = simulation.run()
        
        if simulation.t-tlast_flow > dt_flow:
            dt_act = simulation.t-tlast_flow
            tlast_flow = np.copy(simulation.t)
            R = np.random.random_sample(simulation.lattice.height)
            simulation.perform_stat()
            for iy in range(2,simulation.lattice.height-1):
                if R[iy]<simulation.flow.velocity_field[0,iy,0]*dt_act:
                    cnt += 1
                    X = []
                    O = []
                    for ix in range(simulation.lattice.width):
                        if not simulation.lattice._is_empty(ix,iy):
                            X.append(ix)
                            O.append(simulation.lattice.orientation_map[iy,ix])
                    for x in range(len(X)):
                        simulation.lattice.remove_particle(X[x],iy)
                    for x in range(len(X)):
                        simulation.add_particle((X[x]+1)%simulation.lattice.width, iy, O[x])
                    simulation.stat_flux_counter[iy] += len(X)
            simulation.initialize_rates()
        
        if simulation.t-tlast_dump > dt_dump:
            tlast_dump = np.copy(simulation.t)
            print(simulation.lattice.visualize_lattice())
            print("t = %f, Performed %d shifts" % (simulation.t,cnt))
            cnt = 0
            data_collector.collect_snapshot()
            simulation.dump_stat(fname_stats)

    print(simulation.lattice.visualize_lattice())
    print("t = %f, Performed %d shifts" % (simulation.t,cnt))
    data_collector.collect_snapshot()
    simulation.dump_stat(fname_stats)
    data_exporter.export_data()
            
if __name__ == "__main__":
    main()
