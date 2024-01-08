import torch
import numpy as np
from typing import Optional, Tuple
from lvmc.core.particle_lattice import ParticleLattice, Orientation
from lvmc.core.magnetic_field import MagneticField
from enum import Enum, auto
from typing import NamedTuple


class EventType(Enum):
    REORIENTATION_UP = Orientation.UP.value
    REORIENTATION_LEFT = Orientation.LEFT.value
    REORIENTATION_DOWN = Orientation.DOWN.value
    REORIENTATION_RIGHT = Orientation.RIGHT.value
    MIGRATION = auto()
    # Future event types can be added here, e.g., TRANSPORT_BY_FLOW = auto()


class Event(NamedTuple):
    etype: EventType
    x: int
    y: int

    def is_reorientation(self) -> bool:
        return self.etype in {
            EventType.REORIENTATION_UP,
            EventType.REORIENTATION_LEFT,
            EventType.REORIENTATION_DOWN,
            EventType.REORIENTATION_RIGHT,
        }

    def is_migration(self) -> bool:
        return self.etype == EventType.MIGRATION


class Simulation:
    def __init__(self, g: float, v0: float, **lattice_params: dict) -> None:
        """
        Initialize the simulation with a given lattice, magnetic field, and parameters.

        :param g: Alignment sensitivity parameter.
        :param v0: Base transition rate for particle movement.
        :param lattice_params: Parameters for the lattice.
        """
        self.lattice = ParticleLattice(**lattice_params)
        self.magnetic_field = MagneticField()
        self.g = g
        self.v0 = v0
        self.rates = torch.zeros(
            (len(Orientation) + 1, self.lattice.height, self.lattice.width),
            dtype=torch.float32,
        )
        self.update_rates()
        # Initialize time
        self.t = 0.0

    def update_rates(self):
        """
        Update the rates tensor based on the current state of the lattice.
        """
        n_orientations = len(Orientation)
        TR = self.lattice.compute_tr(self.g)
        self.rates[
            :n_orientations, :, :
        ] = TR  # 4 is the number of reorientation events

        TM = self.lattice.compute_tm(self.v0)
        self.rates[n_orientations, :, :] = TM  # 4 is the number of reorientation events

    def next_event_time(self) -> float:
        """
        Compute the time until the next event.

        :return: float - The time until the next event.
        """
        total_rate = self.rates.sum()
        assert (
            total_rate > 0
        ), "Total rate must be positive to sample from Exponential distribution."
        return torch.distributions.Exponential(total_rate).sample().item()

    def choose_event(self) -> Event:
        """
        Select and return an event based on the current rate distribution.

        This method computes the total rate of events across the lattice and uses
        a probabilistic approach to select an event. The event is chosen based on
        the rates of all possible events (like reorientation or migration), considering
        their respective probabilities. The chosen event's type and coordinates (x, y)
        are returned as an Event object.

        :return: An Event object representing the chosen event, which includes the
                event type and the (x, y) coordinates on the lattice where it occurs.
        """
        rates_flat: torch.Tensor = self.rates.view(-1)
        total_rate: float = rates_flat.sum().item()
        if total_rate == 0:
            raise ValueError(
                "Cannot choose event: Total rate is zero, indicating no possible events."
            )

        chosen_index: int = torch.multinomial(rates_flat, 1).item()

        # Convert the flat index back into 3D index
        # convert the flat index back into 3D index using numpy.unravel_index because torch.unravel_index is not implemented yet

        event_type_index, y, x = np.unravel_index(chosen_index, self.rates.shape)

        event_type = EventType(event_type_index)

        return Event(event_type, x, y)

    def perform_event(self, event: Event) -> None:
        """
        Execute the specified event on the lattice.

        This method determines the type of the given event (reorientation or migration)
        and performs the corresponding action on the lattice. In case of a reorientation
        event, it reorients the particle at the specified location. For a migration event,
        it moves the particle to a new location.

        :param event: The event object to be executed. It contains the event type and
                    the coordinates (x, y) on the lattice where the event occurs.
        """
        if event.is_reorientation():
            orientation = Orientation(event.etype.value)
            self.lattice.reorient_particle(event.x, event.y, orientation)
        elif event.is_migration():
            self.lattice.move_particle(event.x, event.y)
        else:
            raise ValueError(f"Unrecognized event type: {event.etype}")

    def run(self) -> Event:
        """
        Run the simulation for a single time step.

        :return: An Optional tuple (event_type, x, y) representing the event, or None.
        """
        delta_t = self.next_event_time()
        self.t += delta_t
        event = self.choose_event()
        self.perform_event(event)
        self.update_rates()
        return event

    def apply_magnetic_field(self) -> None:
        """
        Apply the magnetic field to the lattice.
        """
        self.magnetic_field.apply(self.lattice)

    def get_magnetic_field_state(self) -> int:
        """
        Get the current direction of the magnetic field.

        :return: int - The current direction of the magnetic field.
        """
        return self.magnetic_field.get_current_direction()
