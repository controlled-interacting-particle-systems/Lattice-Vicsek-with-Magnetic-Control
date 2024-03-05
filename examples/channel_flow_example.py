from lvmc.core.simulation import Simulation
from tqdm import tqdm
from utils import *
from rich import print

# Parameters for ParticleLattice

width = 50
height = 25
g = 1.5
v0 = 100.0
density = 0.3
flow_params = {"type": "poiseuille", "v1": 15}

def main():
    # Initialize the Simulation
    simulation = Simulation(g, v0, width=width, height=height, density=density, flow_params=flow_params)

    n_steps = int(1e6)  # Number of steps to run the simulation for
    obstacles = torch.zeros((height, width), dtype=torch.bool)
    obstacles[0, :] = True
    obstacles[-1, :] = True
    simulation.lattice.set_obstacles(obstacles)

    for _ in (range(n_steps)):
        event = simulation.run()
        if _ % 100 == 0:
            print(simulation.lattice.visualize_lattice())


if __name__ == "__main__":
    main()
