import torch
from vlmc.core.particle_lattice import ParticleLattice
from vlmc.core.magnetic_field import MagneticField
from vlmc.core.simulation import Simulation
from tqdm import tqdm
from examples.utils import *
import numpy as np

#chiara: added seed
seed = 318
torch.manual_seed(seed)
np.random.seed(seed)


def main():
    # Parameters for ParticleLattice
    height, width = 20, 20
    obstacle_topology, sink_topology = "none", "central"
    obstacle_walls, sink_walls = ["top", "left","bottom", "right"], None
    obstacles, sinks = generate_lattice_topology(
        height, width, obstacle_topology, sink_topology, obstacle_walls, sink_walls
    )

    lattice_params = {
        # include relevant parameters here
        "width": width,
        "height": height,
        "density": 0.3,
       # "obstacles": None,
       # "sinks": None,
    }

    # Creating a ParticleLattice instance
    Lattice = ParticleLattice(**lattice_params) #CHIARA: the lattice is initialized in SImulation? I modified that. 
    #Lattice = 0
    
    # Creating a MagneticField instance
    magnetic_field = MagneticField()  # Add parameters if needed

    # Simulation parameters
    g = 2.0  # Alignment sensitivity
    v0 = 100.0  # Base transition rate
    magnetic_field_interval = 10.0  # Time interval for magnetic field application

    # Initialize the Simulation
    simulation = Simulation(Lattice, g, v0, magnetic_field_interval, **lattice_params)
    #simulation = Simulation(g, v0, magnetic_field_interval, **lattice_params)

    n_steps = int(10e5) # Number of steps to run the simulation for
    order_params = np.zeros(n_steps)
    
    print("Initial lattice")
    print(simulation.lattice)
    print(Lattice)
    print(f"Initial order parameter: {simulation.lattice.compute_order_parameter()}")
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

    fileObject = open(f"data/order_params_size{height}{width}_seed{seed}_{g}.dat", "w")
    for _ in tqdm(range(n_steps)):  
        if _ %100==0:
            print(simulation.lattice)
            order_params[_] = simulation.lattice.compute_order_parameter()
            #print(f"Event occurred: particle at {event[0], event[1]} {event_names[event[2]]}, at time {np.round(simulation.t}, 2)")
            array_to_save = np.array([_,simulation.t, order_params[_]]) #chiara: added        
            np.savetxt(fileObject, array_to_save.reshape(1,3), fmt='%g', delimiter = ' ')
            #np.savetxt(fileObject, '\n')
            fileObject.flush()
        
        
    
        #    print('simulation grid \n', sum(simulation.lattice.particles))
        #    print('particle file grid \n', sum(Lattice.particles))
            #print('particle file grid \n', ParticleLattice)

        event = simulation.run()
        
        
    print("Final lattice")
    print(simulation.lattice.griglia)
    print(f"Final order parameter: {simulation.lattice.compute_order_parameter()}")

if __name__ == "__main__":
    main()
