from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from lvmc.data_handling.data_exporter import DataExporter
from tqdm import tqdm
from utils import *
from rich import print

# Parameters for ParticleLattice

width = 80
height = 24
g = 1.5
v0 = 0.0
density = 0.3
flow_params = {"type": "poiseuille", "v1": 1000.0}

def main():
    # Initialize the Simulation
    simul = Simulation(g, v0, width=width, height=height, density=density, flow_params=flow_params)
    n_steps = int(1e5)  # Number of steps to run the simulation for
    obstacles = torch.zeros((height, width), dtype=torch.bool)
    obstacles[0, :] = True
    obstacles[-1, :] = True
    simul.lattice.set_obstacles(obstacles)

    for _ in tqdm(range(n_steps)):
        event = simul.run()
        if _ % 100 == 0:
            print(simul.lattice.visualize_lattice())


if __name__ == "__main__":
    main()
