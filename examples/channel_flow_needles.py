from lvmc.core.simulation import Simulation
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

    n_steps = int(1e6)  # Number of steps to run the simulation for
    obstacles = torch.zeros((height, width), dtype=torch.bool)
    obstacles[0, :] = True
    obstacles[-1, :] = True
    simulation.lattice.set_obstacles(obstacles)

    dt_flow = 0.1/flow_params["v1"]
    tlast = 0
    while simulation.t < tmax:
        event = simulation.run()
        #print("%f %f" % (simulation.t,tlast))
        if simulation.t-tlast > dt_flow:
            dt_act = simulation.t-tlast
            tlast = np.copy(simulation.t)
            print(simulation.lattice.visualize_lattice())
            R = np.random.random_sample(simulation.lattice.height)
            Yshft = []
            for iy in range(2,simulation.lattice.height-1):
                if R[iy]<simulation.flow.velocity_field[0,iy,0]*dt_act:
                    Yshft.append(iy)
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
            simulation.initialize_rates()
            print("t = %f, dt = %f, Performed %d shifts" % (simulation.t,dt_act,len(Yshft)))
            print(Yshft)
            
if __name__ == "__main__":
    main()
