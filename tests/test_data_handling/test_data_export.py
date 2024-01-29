from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector
from lvmc.data_handling.data_exporter import DataExporter

lattice_params = {
    # include relevant parameters here
    "width": 50,
    "height": 25,
    "density": 0.3,
}

# Simulation parameters
g = 2.0  # Alignment sensitivity
v0 = 100.0  # Base transition rate


def test_basic_data_export():
    simulation = Simulation(g, v0, **lattice_params)
    data_collector = DataCollector(simulation)
    data_exporter = DataExporter("simulation_data.hdf5", data_collector)

    n_steps = int(2)  # Number of steps to run the simulation for

    for _ in range(n_steps):
        event = simulation.run()
        data_collector.collect_event(event)
        data_collector.collect_snapshot()

    # Export the collected data to an HDF5 file at the end of the simulation
    data_exporter.export_data()
