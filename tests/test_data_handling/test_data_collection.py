from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_collector import DataCollector

lattice_params = {
    # include relevant parameters here
    "width": 50,
    "height": 25,
    "density": 0.3,
}

# Simulation parameters
g = 2.0  # Alignment sensitivity
v0 = 100.0  # Base transition rate


def test_basic_data_collection():
    simulation = Simulation(g, v0, **lattice_params)
    data_collector = DataCollector(simulation)

    n_steps = int(2)  # Number of steps to run the simulation for

    for _ in range(n_steps):
        event = simulation.run()
        data_collector.collect_event(event)
        data_collector.collect_snapshot()
