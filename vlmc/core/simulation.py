import torch
import numpy as np
from typing import Optional, Tuple
from vlmc.core.particle_lattice import ParticleLattice
from vlmc.core.magnetic_field import MagneticField


class Simulation: #CHIARA: not sure that this is ok. I added lattice as an argument, with self.lattice the attribute of simulation. 
    def __init__(self, Lattice, g, v0, v1, magnetic_field_interval, **lattice_params):
        """
        Initialize the simulation with a given lattice, magnetic field, and parameters.

        :param g: Alignment sensitivity parameter.
        :param v0: Base transition rate for particle movement.
        :param magnetic_field_interval: Time interval to apply the magnetic field.
        :param lattice_params: Parameters for the lattice.
        """
        #self.lattice = ParticleLattice(**lattice_params)
        self.lattice = Lattice
        self.magnetic_field = MagneticField(0)
        self.g = g
        self.v0 = v0
        self.v1 = v1
        self.magnetic_field_interval = magnetic_field_interval
        num_event_types = self.lattice.NUM_ORIENTATIONS + 1
        if self.v1!=0:
            num_event_types+=1
            
        self.rates = torch.zeros(
            (num_event_types, self.lattice.height, self.lattice.width),
            dtype=torch.float32,
        )
        self.time_since_last_magnetic_field = 0.0
        self.update_rates()
        # Initialize time and time till next magnetic field application
        self.t = 0.0
        self.t_magnetic_field = self.magnetic_field_interval

    def update_rates(self):
        """
        Update the rates tensor based on the current state of the lattice.
        """
        TR = self.lattice.compute_tr(self.g, self.v1)
        self.rates[:4, :, :] = TR  # 4 is the number of reorientation events

        TM = self.lattice.compute_tm(self.v0)
        self.rates[4, :, :] = TM  # 4 is the number of migration event

        if self.v1 != 0:
            TF = self.lattice.compute_tf(self.v1)
            self.rates[5, :, :] = TF  # 5 is the number of flow

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

        #chosen_index: int = torch.multinomial(rates_flat / total_rate, 1).item()
        chosen_index: int = torch.multinomial(rates_flat, 1).item() #MODIFIED

        # Convert the flat index back into 3D index
        # convert the flat index back into 3D index using numpy.unravel_index because torch.unravel_index is not implemented yet

        event_type, y, x = np.unravel_index(chosen_index, self.rates.shape) 

        #print('choosen event %d for x=%d,y=%d' %(event_type.item(), x.item(), y.item()))

        return (x.item(), y.item(), event_type.item())

    def perform_event(self, event: Tuple[int, int, int]) -> None:
        """
        Perform an event on the lattice.

        :param event: The event to perform, given as (event_type, x, y)
        """
        x, y, event_type = event

        if event_type < 4:  # Reorientation event
            self.lattice.reorient_particle(x, y, event_type)            
        else:  # Migration event or move with the flow
            if event_type == 4:            
                self.lattice.move_particle(x, y)
            elif event_type == 5:
                self.lattice.move_particle(x, y, True)
            else:
                raise ValueError("Eventtype not defined" )


    def run(self) -> Optional[Tuple[int, int, int]]:
        """
        Run the simulation for a single time step.

        :return: An Optional tuple (event_type, x, y) representing the event, or None.
        """
        self.apply_magnetic_field_if_needed()
        delta_t = self.compute_and_update_time()
        event = self.choose_and_perform_event()
        self.update_rates()
        return event

    def compute_and_update_time(self) -> float:
        """
        Compute the time until the next event and update the simulation time.

        :return: The time delta for the next event.
        """
        delta_t = self.next_event_time()
        self.t += delta_t
        self.time_since_last_magnetic_field += delta_t
        return delta_t

    def apply_magnetic_field_if_needed(self):
        """
        Apply the magnetic field to the lattice if the interval has been reached.

        :param delta_t: The time delta for the next event.
        """
        if self.test_apply_magnetic_field():            
            self.magnetic_field.apply(self.lattice)
            self.time_since_last_magnetic_field = 0.0
            self.update_rates()

    def choose_and_perform_event(self) -> Optional[Tuple[int, int, int]]:
        """
        Choose and perform an event.

        :return: An Optional tuple representing the event, or None.
        """
        event = self.choose_event()
        self.perform_event(event)
        return event

    def update_simulation_state(self, delta_t: float):
        """
        Update the rates and the time till the next magnetic field application.

        :param delta_t: The time delta for the next event.
        """
        self.update_rates()
        self.t_magnetic_field -= delta_t

    def test_apply_magnetic_field(self) -> bool:        
        return (bool(self.time_since_last_magnetic_field >= self.magnetic_field_interval))
