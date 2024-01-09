from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from lvmc.data_handling.visualization import Visualization
from parameters import g, v0, lattice_params, obstacles, sinks
from tqdm import tqdm
import matplotlib.pyplot as plt


def run_simulation():
    simulation = Simulation(g, v0, **lattice_params)
    data_collector = DataCollector(simulation)
    visualizer = Visualization(data_collector)

    plt.ion()  # Turn on interactive mode for real-time updating

    n_steps = int(1e4)  # Number of steps to run the simulation for

    for step in tqdm(range(n_steps)):
        event = simulation.run()
        data_collector.collect_event(event)
        data_collector.collect_snapshot()

        # Fetch the latest snapshot for visualization
        latest_snapshot = data_collector.data["snapshots"][-1][1]
        visualizer.update_plot(latest_snapshot)
        plt.draw()
        plt.pause(0.00001)  # Pause to update the figure

    plt.ioff()  # Turn off interactive mode


if __name__ == "__main__":
    run_simulation()
