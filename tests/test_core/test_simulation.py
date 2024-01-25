from lvmc.core.simulation import Simulation

lattice_params = {
    # include relevant parameters here
    "width": 50,
    "height": 25,
    "density": 0.3,
}

# Simulation parameters
g = 2.0  # Alignment sensitivity
v0 = 100.0  # Base transition rate


def test_basic_simulation():
    simulation = Simulation(g, v0, **lattice_params)
    n_steps = int(1e2)  # Number of steps to run the simulation for
    for _ in range(n_steps):
        event = simulation.run()