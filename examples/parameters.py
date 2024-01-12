# Parameters for ParticleLattice
lattice_params = {
    # include relevant parameters here
    "width": 100,
    "height": 10,
    "density": 0.3,
}

# Simulation parameters
g = 2.0  # Alignment sensitivity
v0 = 100.0  # Base transition rate


# Topology setup
from utils import *

sink_walls = "right", "left"
obstacle_walls = "top", "bottom"
obstacles, sinks = generate_lattice_topology(
    lattice_params["height"],
    lattice_params["width"],
    sink_walls=sink_walls,
    obstacle_walls=obstacle_walls,
)

print(obstacles.shape)
