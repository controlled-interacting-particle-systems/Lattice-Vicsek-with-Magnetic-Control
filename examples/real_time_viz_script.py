from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from lvmc.data_handling.visualization import Visualization
from parameters import *
from tqdm import tqdm
import matplotlib.pyplot as plt


def run_simulation():
    simulation = Simulation(g, v0, **lattice_params)
    simulation.lattice.set_obstacles(obstacles)
    data_collector = DataCollector(simulation)
    visualizer = Visualization(data_collector)



    n_steps = int(1e7)  # Number of steps to run the simulation for
    update_freq = 1000  # Number of steps between each visualization update

    for step in tqdm(range(n_steps)):
        event = simulation.run()
        if step % update_freq == 0:
            data_collector.collect_event(event)
            data_collector.collect_snapshot()

            # Fetch the latest snapshot for visualization
            latest_snapshot = data_collector.data["snapshots"][-1][1]
            visualizer.update_plot(latest_snapshot)
            plt.draw()
            plt.pause(0.000000001)  # Pause to update the figure



if __name__ == "__main__":
    run_simulation()
