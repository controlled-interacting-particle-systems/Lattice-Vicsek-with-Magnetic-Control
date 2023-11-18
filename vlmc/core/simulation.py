import torch
import numpy as np
from typing import Optional, Tuple
from vlmc.core.particle_lattice import ParticleLattice
from vlmc.core.magnetic_field import MagneticField


class Simulation:
    def __init__(self, lattice, magnetic_field, g, v0, magnetic_field_interval):
        """
        Initialize the simulation with a given lattice, magnetic field, and parameters.

        :param lattice: An instance of the ParticleLattice class.
        :param magnetic_field: An instance of the MagneticField class.
        :param g: Alignment sensitivity parameter.
        :param v0: Base transition rate for particle movement.
        :param magnetic_field_interval: Time interval to apply the magnetic field.
        """
        self.lattice = lattice
        self.magnetic_field = magnetic_field
        self.g = g
        self.v0 = v0
        self.magnetic_field_interval = magnetic_field_interval
        self.rates = torch.zeros(
            (5, lattice.height, lattice.width), dtype=torch.float32
        ) # 5 is the number of event types (4 reorientation events and 1 migration event) could be a variable called num_event_types
        self.time_since_last_magnetic_field = 0.0
        self.update_rates()
        # Initialize time and time till next magnetic field application
        self.t = 0.0
        self.t_magnetic_field = self.magnetic_field_interval

    def update_rates(self):
        """
        Update the rates tensor based on the current state of the lattice.
        """
        TR = self.lattice.compute_TR(self.g)
        self.rates[:4, :, :] = TR # 4 is the number of reorientation events

        TM = self.lattice.compute_TM(self.v0)
        self.rates[4, :, :] = TM # 4 is the number of reorientation events

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
            return None

        chosen_index: int = torch.multinomial(rates_flat / total_rate, 1).item()

        # Convert the flat index back into 3D index
        # convert the flat index back into 3D index using numpy.unravel_index because torch.unravel_index is not implemented yet

        event_type, y, x = np.unravel_index(chosen_index, self.rates.shape)

        return (event_type.item(), y.item(), x.item())

    def perform_event(self, event: Tuple[int, int, int]) -> None:
        """
        Perform an event on the lattice.

        :param event: The event to perform, given as (event_type, x, y)
        """
        event_type, y, x = event

        if event_type < 4:  # Reorientation event
            self.lattice.reorient_particle(x, y, event_type)
        else:  # Migration event
            self.lattice.move_particle(x, y)

    def run(self) -> Optional[Tuple[int, int, int]]:
        """
        Run the simulation for a single time step.

        :return: An Optional tuple (event_type, x, y) representing the event, or None.
        """
        # Compute the time until the next event
        delta_t = self.next_event_time()
        # Update the time
        self.t += delta_t
        self.time_since_last_magnetic_field += delta_t

        # Check if it is time to apply the magnetic field
        if self.time_since_last_magnetic_field >= self.magnetic_field_interval:
            self.magnetic_field.apply(self.lattice)
            self.time_since_last_magnetic_field = 0.0

        # Choose an event
        event = self.choose_event()

        # Perform the event
        self.perform_event(event)

        # Update the rates
        self.update_rates()

        # Update the time till next magnetic field application
        self.t_magnetic_field -= delta_t

        # Return the event
        return event
