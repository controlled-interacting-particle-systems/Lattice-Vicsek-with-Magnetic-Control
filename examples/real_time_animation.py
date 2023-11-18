# Import necessary modules
from vlmc.core.simulation import Simulation
from vlmc.data_handling.visualization import Visualization

# Initialize simulation parameters
width, height = 10, 10  # Lattice dimensions
v0 = 100.0  # Base rate for migration
g = 2  # Alignment sensitivity parameter
density = 0.3  # Initial density of particles

# Initialize the simulation
sim = Simulation(width, height, v0, g, density)

# Initialize the visualization
vis = Visualization(sim.lattice)

# Run the animation
num_frames = 10000  # Number of frames
time_interval = 1  # Time interval between frames in milliseconds

vis.animate_lattice(num_frames, time_interval, sim)
