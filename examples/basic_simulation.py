import torch
from lvmc.core.particle_lattice import ParticleLattice
from lvmc.core.magnetic_field import MagneticField
from lvmc.core.simulation import Simulation
from tqdm import tqdm
from examples.utils import *
import numpy as np


def main():
    # Parameters for ParticleLattice
    height, width = 10, 10
    obstacle_topology, sink_topology = "none", "central"
    obstacle_walls, sink_walls = ["top", "left", "bottom", "right"], None
    obstacles, sinks = generate_lattice_topology(
        height, width, obstacle_topology, sink_topology, obstacle_walls, sink_walls
    )

    lattice_params = {
        # include relevant parameters here
        "width": width,
        "height": height,
        "density": 0.3,
        "obstacles": None,
        "sinks": None,
    }

    # Creating a ParticleLattice instance
    lattice = ParticleLattice(**lattice_params)

    # Creating a MagneticField instance
    magnetic_field = MagneticField()  # Add parameters if needed

    # Simulation parameters
    g = 2.0  # Alignment sensitivity
    v0 = 100.0  # Base transition rate
    magnetic_field_interval = 10.0  # Time interval for magnetic field application

    # Initialize the Simulation
    simulation = Simulation(g, v0, magnetic_field_interval, **lattice_params)

    n_steps = int(10000)  # Number of steps to run the simulation for


    print("Initial lattice")
    print(lattice)
    
    # create a mapping between the event codes and the event names for printing purposes
    # 0 is up 1 is left 2 is down 3 is right 4 is migrate
    event_names = [
        "turned up",
        "turned left",
        "turned down",
        "turned right",
        "migrated to the next cell",
    ]
    event_codes = [0, 1, 2, 3, 4]
    event_map = dict(zip(event_codes, event_names))

    for _ in tqdm(range(n_steps)):
        event = simulation.run()

    print("Final lattice")
    print(lattice)

if __name__ == "__main__":
    main()
