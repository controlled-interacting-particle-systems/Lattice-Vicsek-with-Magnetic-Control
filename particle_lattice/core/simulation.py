import torch

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

        """
        Run the simulation for a single time step.
        :param delta_t: The time increment for the simulation step.
        """
        pass
