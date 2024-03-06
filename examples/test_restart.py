from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from lvmc.data_handling.data_exporter import DataExporter
from tqdm import tqdm
from utils import *
from rich import print

# Parameters for ParticleLattice

width = 20
height = 11
g = 1.5
v0 = 100.0
density = 0.3
flow_params = {"type": "poiseuille", "v1": 15}

def main():
    # Initialize the Simulation
    simul1 = Simulation(g, v0, width=width, height=height, density=density, flow_params=flow_params)
    n_steps = int(1000)  # Number of steps to run the simulation for
    obstacles = torch.zeros((height, width), dtype=torch.bool)
    obstacles[0, :] = True
    obstacles[-1, :] = True
    simul1.lattice.set_obstacles(obstacles)

    data_collector = DataCollector(simul1)
    fname = "test_io.hdf5"
    data_exporter = DataExporter(fname, data_collector)

    print("Starting simulation 1 at time %f" % simul1.t)
    for _ in tqdm(range(n_steps)):
        event = simul1.run()
        data_collector.collect_event(event)
        if _ % 100 == 0:
            data_collector.collect_snapshot()
    print(simul1.lattice.visualize_lattice())
    print("Finishing simulation 1 at time %f" % simul1.t)
    print("and writing data")    
    data_exporter.export_data()

    print("Initializing simulation 2")
    simul2 = Simulation.init_from_file(fname)
    print("Ready to start at time %f" % simul2.t)
    print(simul2.lattice.visualize_lattice())
    

if __name__ == "__main__":
    main()
