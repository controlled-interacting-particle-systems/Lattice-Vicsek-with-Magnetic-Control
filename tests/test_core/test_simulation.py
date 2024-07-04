from lvmc.core.simulation import Simulation
import pytest
import torch


# Define the parameters
g = 1.0
v0 = 100.0
width = 10
height = 5
density = 0.3
flow_params = {"type": "Poiseuille", "v1": 100}
obstacles = torch.zeros((height, width), dtype=torch.bool)
obstacles[0, :] = True
obstacles[-1, :] = True


class TestSimulation:
    @pytest.fixture
    def simulation(self):
        simulation = (
            Simulation(g, v0)
            .add_lattice(width=width, height=height)
            .add_flow(flow_params)
            .add_obstacles(obstacles)
            .add_particles(density=density)
            .build()
        )
        return simulation

    def test_choose_event(self, simulation):
        event = simulation.choose_event()
        # check that the rate for the chosen event is non-zero
        simulation.rates[event.etype.value, event.y, event.x] > 0

    def test_perform_event(self, simulation):
        event = simulation.choose_event()
        simulation.perform_event(event)
        # check that the number of particles is conserved
