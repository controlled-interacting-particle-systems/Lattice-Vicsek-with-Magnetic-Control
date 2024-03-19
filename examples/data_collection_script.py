# data_collection_script.py

from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from tqdm import tqdm
from parameters import flow_params, g, v0, width, height, density, obstacles


def run_simulation():
    simulation = (
        Simulation(g, v0)
        .add_lattice(width=width, height=height)
        .add_flow(flow_params)
        .add_obstacles(obstacles)
        .add_particles(density=density)
        .add_control_field()
        .build()
    )
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
