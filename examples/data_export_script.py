from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from lvmc.data_handling.data_exporter import DataExporter
from parameters import g, v0, width, height, density, flow_params, obstacles
from tqdm import tqdm


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
    data_exporter = DataExporter("simulation_data.hdf5", data_collector)

    n_steps = int(1e4)  # Number of steps to run the simulation for

    for _ in tqdm(range(n_steps)):
        event = simulation.run()
        data_collector.collect_event(event)
        data_collector.collect_snapshot()

    # Export the collected data to an HDF5 file at the end of the simulation
    data_exporter.export_data()


if __name__ == "__main__":
    run_simulation()
