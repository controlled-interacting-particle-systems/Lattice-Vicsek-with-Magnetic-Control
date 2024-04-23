from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from lvmc.data_handling.data_exporter import DataExporter
from tqdm import tqdm
from utils import *
from rich import print
import numpy as np
import sys

# Parameters for ParticleLattice

seed = 3179
torch.manual_seed(seed)
np.random.seed(seed)

height = int(sys.argv[1])
width = int(sys.argv[2])
v0 = float(sys.argv[3])
density = 0.3
flow_params = {"type": "poiseuille", "v1": float(sys.argv[4])}
tmax = 500
t0 = 0
    
def main(g: float = float(sys.argv[5])):
    if flow_params["v1"] == 0.:
        dt_flow = 2*tmax
        base_name = "noflow_"+str(width)+"_"+str(height)+"_"+str(g)+"_"+str(v0).removesuffix('.0')
    else: 
        dt_flow = 0.1/flow_params["v1"]
        base_name = flow_params["type"]+"_"+str(width)+"_"+str(height)+"_"+str(g)+"_"+str(v0).removesuffix('.0')+"_"+str(flow_params["v1"]).removesuffix('.0')
    fname_stats = "file_output/" + "stat_"+base_name+ "seed%d"%(seed) + ".txt"
    dt_stat = 0.1
    dt_dump_stat = 5
    dt_dump_field = 50
    
    # Initialize the Simulation (test if restart of not)
    if t0==0:
        print("Starting simulation from scratch with g = %g, v1 = %g" % (g,flow_params["v1"]))
        simulation = Simulation(g, v0, width=width, height=height, density=density, flow_params=flow_params, with_transport=False)
        obstacles = torch.zeros((height, width), dtype=torch.bool)
        obstacles[0, :] = True
        obstacles[-1, :] = True
        simulation.lattice.set_obstacles(obstacles)
    else:
        fname =  "file_output/" + "fields_"+base_name+"_"+("%1.2f"%(t0-dt_dump_field))+"_"+("%1.2f"%t0)+ "seed%d"%(seed) +".h5"
        simulation = Simulation.init_from_file(fname)
    data_collector = DataCollector(simulation)
    
    tlast_flow = t0
    count_flow = int(t0/dt_flow)
    count_stat = int(t0/dt_stat)
    count_dump_stat = int(t0/dt_dump_stat)
    tlast_dump_field = t0
    count_dump_field = int(t0/dt_dump_field)
    simulation.init_stat()
    cnt = 0
    
    while simulation.t < tmax:
        event = simulation.run()
        
        if simulation.t-count_flow*dt_flow > dt_flow:
            dt_act = simulation.t-tlast_flow
            tlast_flow = np.copy(simulation.t)
            count_flow += 1
            Nshift = np.random.poisson(simulation.flow.velocity_field[0,1:-1,0]*dt_act)
            for iy in range(1,simulation.lattice.height-1):
                if Nshift[iy-1]>0:
                    cnt += Nshift[iy-1]
                    X = np.argwhere(simulation.lattice.occupancy_map[iy,:]).squeeze().tolist()
                    O = simulation.lattice.orientation_map[iy,X]
                    if isinstance(X,int):
                        simulation.lattice.remove_particle(X,iy)
                        simulation.add_particle((X+Nshift[iy-1])%simulation.lattice.width, iy, O)
                        simulation.stat_flux_counter[iy] += Nshift[iy-1]
                    else:
                        for x in X:
                            simulation.lattice.remove_particle(x,iy)
                        for ix in range(len(X)):
                            simulation.add_particle((X[ix]+Nshift[iy-1])%simulation.lattice.width, iy, O[ix])
                        simulation.stat_flux_counter[iy] += Nshift[iy-1]*len(X)
            simulation.initialize_rates()
        
        if simulation.t-count_stat*dt_stat > dt_stat:
            count_stat += 1
            simulation.perform_stat()
            #print(simulation.lattice.visualize_lattice())
            print("t = %f, Performed %d shifts" % (simulation.t,cnt))
            cnt = 0
            data_collector.collect_snapshot()
            
        if simulation.t-count_dump_stat*dt_dump_stat > dt_dump_stat:
            count_dump_stat += 1
            simulation.dump_stat(fname_stats)
            
        if simulation.t-count_dump_field*dt_dump_field > dt_dump_field:
            fname_dumps = "file_output/" +"fields_"+base_name+"_"+("%1.2f"%tlast_dump_field)+"_"+("%1.2f"%simulation.t)+"seed%d"%(seed) +".h5"
            count_dump_field += 1
            tlast_dump_field = np.copy(simulation.t)
            data_exporter = DataExporter(fname_dumps, data_collector)
            data_exporter.export_data()
            data_collector = DataCollector(simulation)
    
    #print(simulation.lattice.visualize_lattice())
    print("t = %f, Performed %d shifts" % (simulation.t,cnt))
    data_collector.collect_snapshot()
    simulation.dump_stat(fname_stats)
    fname_dumps = "file_output/" + "fields_"+base_name+"_"+("%1.2f"%tlast_dump_field)+"_"+("%1.2f"%simulation.t)+"seed%d"%(seed) +".h5"
    data_exporter = DataExporter(fname_dumps, data_collector)
    data_exporter.export_data()
            
if __name__ == "__main__":
    if len(sys.argv)>1:
        main(float(sys.argv[5]))
    else:
        main()
