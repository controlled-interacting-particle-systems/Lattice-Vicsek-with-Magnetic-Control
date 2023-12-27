import torch
import numpy as np
from typing import Optional, Tuple
from lvmc.core.particle_lattice import ParticleLattice
from lvmc.core.magnetic_field import MagneticField


class Simulation:
    def __init__(self, g, v0, **lattice_params):
        """
        Initialize the simulation with a given lattice, magnetic field, and parameters.

        :param g: Alignment sensitivity parameter.
        :param v0: Base transition rate for particle movement.
        :param magnetic_field_interval: Time interval to apply the magnetic field.
        :param lattice_params: Parameters for the lattice.
        """
        self.lattice = ParticleLattice(**lattice_params)
        self.magnetic_field = MagneticField()
        self.g = g
        self.v0 = v0
        self.num_event_types = self.lattice.NUM_ORIENTATIONS + 1
        self.rates = torch.zeros(
            (self.num_event_types, self.lattice.height, self.lattice.width),
            dtype=torch.float32,
        )
        self.update_rates()
        # Initialize time
        self.t = 0.0

    def update_rates(self):
        """
        Update the rates tensor based on the current state of the lattice.
        """
        TR = self.lattice.compute_tr(self.g)
        self.rates[:4, :, :] = TR  # 4 is the number of reorientation events

        TM = self.lattice.compute_tm(self.v0)
        self.rates[4, :, :] = TM  # 4 is the number of reorientation events

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

    def choose_event(self) -> Optional[Tuple[int, int, int]]:
        """
        Choose an event based on the rates.

        :return: An Optional tuple (event_type, x, y) if an event is chosen, else None.
        """
        rates_flat: torch.Tensor = self.rates.view(-1)
        total_rate: float = rates_flat.sum().item()
        if total_rate == 0:
            raise ValueError(
                "Total rate must be positive to sample from Multinomial distribution."
            )

        chosen_index: int = torch.multinomial(rates_flat, 1).item()

        # Convert the flat index back into 3D index
        # convert the flat index back into 3D index using numpy.unravel_index because torch.unravel_index is not implemented yet

        event_type, y, x = np.unravel_index(chosen_index, self.rates.shape)

        # check if the position is empty and raise a value error if it is
        if self.lattice.is_empty(x, y):
            raise ValueError(f"Position ({x}, {y}) is empty.")

        return (x.item(), y.item(), event_type.item())

    def perform_event(self, event: Tuple[int, int, int]) -> None:
        """
        Perform an event on the lattice.

        :param event: The event to perform, given as (event_type, x, y)
        """
        x, y, event_type = event

        if event_type < self.num_event_types - 1:  # Reorientation event
            return self.lattice.reorient_particle(x, y, event_type)
        else:  # Migration event
            return self.lattice.move_particle(x, y)

    def run(self) -> Optional[Tuple[int, int, int]]:
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
