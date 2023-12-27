from lvmc.core.simulation import Simulation
from tqdm import tqdm
from utils import *
from parameters import *
import cProfile
import pstats


def main():
    # Initialize the Simulation
    simulation = Simulation(g, v0, **lattice_params)
    n_steps = int(1e4)  # Number of steps to run the simulation for

    # Start profiling here
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in tqdm(range(n_steps)):
        event = simulation.run()

    # Stop profiling and save the results
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.dump_stats("simulation_run_profile.prof")


if __name__ == "__main__":
    main()
