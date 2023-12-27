# Import necessary modules
from particle_lattice.core.simulation import Simulation
from particle_lattice.visualization.visualization import Visualization
from tqdm import tqdm

# Initialize simulation parameters
width, height = 50, 25  # Lattice dimensions
v0 = 100.0  # Base rate for migration
g = 2  # Alignment sensitivity parameter
density = 0.3  # Initial density of particles

# Initialize the simulation
sim = Simulation(width, height, v0, g, density)

for _ in tqdm(range(100000)):
    sim.run_time_step()