import torch
from vlmc.core.particle_lattice import ParticleLattice
from vlmc.core.magnetic_field import MagneticField
from vlmc.core.simulation import Simulation
from tqdm import tqdm
from examples.utils import *


def main():
    

    # Parameters for ParticleLattice
    height, width = 100, 100
    obstacle_topology, sink_topology = "none", "central"
    obstacle_walls, sink_walls = ["top", "left"], ["bottom", "right"]
    obstacles, sinks = generate_lattice_topology(height, width, obstacle_topology, sink_topology, obstacle_walls, sink_walls)

    lattice_params = {
        # include relevant parameters here
        "width":width,
        "height":height,
        "density":0.1,
        "obstacles":obstacles,
        "sinks":sinks,
    }

        # Creating a ParticleLattice instance
    lattice = ParticleLattice(**lattice_params)

    # Creating a MagneticField instance
    magnetic_field = MagneticField()  # Add parameters if needed

    # Simulation parameters
    g = 1.0  # Alignment sensitivity
    v0 = 1.0  # Base transition rate
    magnetic_field_interval = 10.0  # Time interval for magnetic field application

    # Set up the simulation
# Initialize the Simulation
    simulation = Simulation(lattice_params, g, v0, magnetic_field, magnetic_field_interval)


    # Run the simulation for a number of steps
    num_steps = 10000
    for step in tqdm(range(num_steps)):
        simulation.run()


if __name__ == "__main__":
    main()
