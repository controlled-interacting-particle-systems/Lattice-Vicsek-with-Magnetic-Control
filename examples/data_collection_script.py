# data_collection_script.py

from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from tqdm import tqdm
from parameters import g, v0, lattice_params


def run_simulation():
    simulation = Simulation(g, v0, **lattice_params)
    data_collector = DataCollector(simulation)

    n_steps = int(2)  # Number of steps to run the simulation for

    for _ in tqdm(range(n_steps)):
        event = simulation.run()
        data_collector.collect_event(event)
        data_collector.collect_snapshot()

    # Here you can add code to inspect the collected data
    print("Collected Data:", data_collector.data)


if __name__ == "__main__":
    run_simulation()
