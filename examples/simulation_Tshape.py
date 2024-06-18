from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from lvmc.data_handling.data_exporter import DataExporter
from parameters_Tshape import *
from tqdm import tqdm
from utils import *
from rich import print
import numpy as np
import sys

def main():
    # Initialize the Simulation
    print(list_part_right)
    simulation = (
        Simulation(g, v0, seed=81238)
        .add_lattice(width=width, height=height)
        .add_obstacles(obstacles)
        .add_sinks(sinks)
        # .add_particles_from_list([],[],[],list_part_right)
        # .build()
    )
    Stat = []
    for ireal in range(1000):
        this_real = []
        simulation.t=0
        simulation.add_particles_from_list([],[],[],list_part_right)
        simulation.build()
        print(simulation.lattice.visualize_lattice())
        n_part = simulation.lattice.particles.sum().item()
        n_part_init = np.copy(n_part)
        n_sink_left = 0
        n_sink_up = 0
        n_sink_down = 0
        it = 0
        while n_part > 0:
            event = simulation.run()
            n_part_new = simulation.lattice.particles.sum().item()
            if n_part_new < n_part:
                if event.x == 1:
                    n_sink_left += 1
                    this_real.extend([np.copy(simulation.t).item(),0])
                elif event.y == 1:
                    n_sink_up += 1
                    this_real.extend([np.copy(simulation.t).item(),1])
                elif event.y == height-2:
                    n_sink_down += 1
                    this_real.extend([np.copy(simulation.t).item(),2])
                else:
                    print(event)
                n_part = np.copy(n_part_new)
            if it%100 == 0:
                print(simulation.lattice.visualize_lattice())
                print('\n')
            it += 1
        print(f"Realization {ireal}, number of particles: {n_part_init}")
        print(f" absorbed by left sink:  {n_sink_left}")
        print(f" absorbed by down sink:  {n_sink_down}")
        print(f" absorbed by upper sink: {n_sink_up}")
        Stat.append(this_real)
    with open("stat_Tshape.txt", "w") as file:
        for sublist in Stat:
            line = ' '.join(map(str, sublist))
            file.write(line + '\n')
    
if __name__ == "__main__":
    main()
