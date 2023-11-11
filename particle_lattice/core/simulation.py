import torch
from typing import Optional, Tuple

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
        self.rates = torch.zeros((5, lattice.height, lattice.width), dtype=torch.float32)
        self.time_since_last_magnetic_field = 0.0
        self.update_rates()

    def update_rates(self):
        """
        Update the rates tensor based on the current state of the lattice.
        """
        TR = self.lattice.compute_TR(self.g)
        self.rates[:4, :, :] = TR

        TM = self.lattice.compute_TM(self.v0)
        self.rates[4, :, :] = TM
    
    def next_event_time(self) -> float:
        """
        Compute the time until the next event.

        :return: float - The time until the next event.
        """
        total_rate = self.rates.sum()
        assert total_rate > 0, "Total rate must be positive to sample from Exponential distribution."
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
        event_type, y, x = torch.unravel_index(chosen_index, self.rates.shape)
        return (event_type.item(), y.item(), x.item())
    
    def perform_event(self, event: Tuple[int, int, int]) -> None:
        """
        Perform an event on the lattice.

        :param event: The event to perform, given as (event_type, x, y)
        """
        event_type, x, y = event

        if event_type < 4:  # Reorientation event
            self.lattice.reorient_particle(x, y, event_type)
        else:  # Migration event
            self.lattice.move_particle(x, y)


        """
        Run the simulation for a single time step.
        :param delta_t: The time increment for the simulation step.
        """
        pass
