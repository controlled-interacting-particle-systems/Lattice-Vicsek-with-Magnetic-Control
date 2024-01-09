# data_collector.py

from lvmc.core.simulation import Simulation, Event


class DataCollector:
    def __init__(self, simulation: 'Simulation') -> None:
        """
        Initialize the DataCollector with a simulation object.

        :param simulation: The simulation object from which data will be collected.
        """
        self.simulation = simulation
        self.data = {
            "metadata": {
                "g": simulation.g,
                "v0": simulation.v0,
                "lattice_params": simulation.lattice.get_params(),  # Assuming a method to fetch lattice parameters
            },
            "initial_config": simulation.lattice.query_lattice_state(),
            "obstacles": simulation.lattice.obstacles,
            "sinks": simulation.lattice.sinks,
            "snapshots": [],
            "events": [],
        }

    def collect_snapshot(self) -> None:
        """
        Collect and store a snapshot of the current state of the simulation.

        The snapshot includes the current time and the state of the lattice.
        """
        self.data["snapshots"].append(
            (self.simulation.t, self.simulation.lattice.query_lattice_state())
        )

    def collect_event(self, event: 'Event') -> None:
        """
        Collect and store data about an event that occurred in the simulation.

        :param event: The event object containing details about the event.
                      Assumes that the event object has attributes etype, x, y.
        """
        if event:
            event_data = {
                "time": self.simulation.t,
                "event_type": event.etype.value,
                "x": event.x,
                "y": event.y,
                "magnetic_field": self.simulation.get_magnetic_field_state(),  # Assuming a method to fetch current magnetic field state
            }
            self.data["events"].append(event_data)
