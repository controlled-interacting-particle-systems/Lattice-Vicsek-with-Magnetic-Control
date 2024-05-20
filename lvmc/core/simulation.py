import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from lvmc.core.particle_lattice import ParticleLattice, Orientation
from lvmc.core.magnetic_field import MagneticField
from lvmc.core.flow import PoiseuilleFlow
from enum import Enum, auto
from typing import NamedTuple, List, Optional
import json
import h5py



class EventType(Enum):
    REORIENTATION_UP = Orientation.UP.value
    REORIENTATION_LEFT = Orientation.LEFT.value
    REORIENTATION_DOWN = Orientation.DOWN.value
    REORIENTATION_RIGHT = Orientation.RIGHT.value
    MIGRATION = auto()
    BIRTH = auto()
    TRANSPORT_UP = auto()
    TRANSPORT_LEFT = auto()
    TRANSPORT_DOWN = auto()
    TRANSPORT_RIGHT = auto()
    # DEATH = auto()
    # Future event types can be added here, e.g., TRANSPORT_BY_FLOW = auto()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def is_birth(self) -> bool:
        return self.etype == EventType.BIRTH

    def is_transport(self) -> bool:
        return self.etype in {
            EventType.TRANSPORT_UP,
            EventType.TRANSPORT_LEFT,
            EventType.TRANSPORT_DOWN,
            EventType.TRANSPORT_RIGHT,
        }


class Simulation:
    def __init__(
        self, g: float, v0: float, width: int, height: int, density: float, flow_params: Optional[dict] = None,
        init_time: Optional[float] = 0.0, init_part: torch.float32 = None, obstacles: torch.float32 = None, with_transport: bool=True) -> None:
        """
        Initialize the simulation with a given lattice, magnetic field, and parameters.

        :param g: Alignment sensitivity parameter.
        :param v0: Base transition rate for particle movement.
        :param lattice_params: Parameters for the lattice.
        """
        self.density = density
        self.lattice = ParticleLattice(width=width, height=height)
        self.with_transport = with_transport

        if not init_part is None:
            if obstacles is None:
                print("Warning: starting with prescribed particles but no obstacle")
            else:
                self.lattice.set_obstacles(obstacles)
            for x in range(width):
                for y in range(height):
                    for ior in list(Orientation):
                        if init_part[ior.value, y, x]:
                            self.lattice.add_particle(x,y,ior)
        elif not obstacles is None:
            print("Warning: starting with prescribed obstacles but no particles")
        self.magnetic_field = MagneticField()
        self.flow_params = flow_params
        if flow_params is None:
            print("Simulation without flow")
            self.with_flow = False
        elif flow_params["type"] == "poiseuille":
            print("Simulation with Poiseuille flow")
            self.flow = PoiseuilleFlow(
                self.lattice.width, self.lattice.height, flow_params["v1"]
            )
            self.with_flow = True
        else:
            raise ValueError(f"Unrecognized flow type: {flow_params['type']}")
        self.g = g
        self.v0 = v0
        self.rates = torch.zeros(
            (len(EventType), self.lattice.height, self.lattice.width),
            dtype=torch.float32,
            device=device,
        )
        self.initialize_rates()
        self.with_stat = False
        # Initialize time
        self.t = init_time
        
    @classmethod
    def init_from_file(cls, fname:str) -> None:
        """
        Initialize the simulation from hdf5 file.

        :param fname: file name
        """
        print("Initializing simulation from file", fname)
        with h5py.File(fname, "r") as file:
            sim_params = json.loads(file.attrs["simulation_params"])
            lattice_params = json.loads(file.attrs["lattice_params"])
            Obs = torch.tensor(np.array(file['obstacles']))
            last_snap = list(file['snapshots'])[-1]
            Part = torch.tensor(np.array(file['snapshots'][last_snap]))
            t0 = file['snapshots'][last_snap].attrs['time']
        NewSim = cls(sim_params["alignment sensitivity"], sim_params["migration rate"], lattice_params["width"],
                     lattice_params["height"], sim_params["density"], sim_params["flow parameters"], init_time=t0, init_part=Part, obstacles=Obs)
        return NewSim

    @classmethod
    def init_config_from_file(cls, fname:str, g: float, v0: float, flow: Optional[dict] = None, t0: Optional[float] = 0.0) -> None:
        """
        Initialize the simulation from hdf5 file.
        
        :param fname: file name
        :param g: Alignment sensitivity parameter.
        :param v0: Base transition rate for particle movement.
        :param flow: Flow type
        :param t0: Initial time
        """
        print("Initializing simulation from file", fname)
        with h5py.File(fname, "r") as file:
            sim_params = json.loads(file.attrs["simulation_params"])
            lattice_params = json.loads(file.attrs["lattice_params"])
            Obs = torch.tensor(np.array(file['obstacles']))
            last_snap = list(file['snapshots'])[-1]
            Part = torch.tensor(np.array(file['snapshots'][last_snap]))
            t0 = file['snapshots'][last_snap].attrs['time']
        NewSim = cls(g, v0, lattice_params["width"], lattice_params["height"], sim_params["density"], flow_params=flow, init_time=t0, init_part=Part, obstacles=Obs)
        return NewSim

    def get_params(self) -> dict:
        """
        Get the parameters of the simulation.
        :return: A dictionary of the simulation parameters.
        """
        return {
            "density": self.density,
            "alignment sensitivity": self.g,
            "migration rate": self.v0,
            "flow parameters": self.flow_params
        }
        
    def add_particle(self, x: int, y: int, orientation: Orientation = None) -> None:
        """
        Add a particle at the specified location.

        :param x: The x-coordinate of the location.
        :param y: The y-coordinate of the location.
        :param orientation: The orientation of the particle.
        """
        self.lattice.add_particle(x, y, orientation)
        self.update_rates()

    def add_particle_flux(
        self,
        region: Tuple[int, int, int, int],
        orientation: Orientation,
        n_particles: int,
    ) -> None:
        """
        Add a particle flux to the lattice.

        :param region: A tuple (x1, y1, x2, y2) representing the region where the particles will be added.
        :param orientation: The orientation of the particles.
        :param n_particles: The number of particles to be added.
        """
        self.lattice.add_particle_flux(region, orientation, n_particles)
        self.update_rates()

    def populate_lattice(self, density: float) -> None:
        """
        Populate the lattice with particles.

        :param density: The density of the particles.
        """
        n_added = self.lattice.populate(density)
        self.update_rates()

    def initialize_rates(self) -> None:
        """
        Initialize the rates tensor.
        """
        n_orientations = len(Orientation)
        self.rates[:n_orientations] = self.lattice.compute_tr(self.g)
        self.rates[EventType.MIGRATION.value] = self.lattice.compute_tm(self.v0)
        self.rates[EventType.BIRTH.value] = self.lattice.compute_birth_rates(self.v0)
        if self.with_flow:
            if self.with_transport:
                self.rates[
                    EventType.TRANSPORT_UP.value : EventType.TRANSPORT_RIGHT.value + 1
                ] = self.flow.compute_tm(self.lattice.occupancy_map)
            self.rates[:n_orientations] += self.flow.compute_tr(self.lattice)

    def update_rates(self, positions: list[Optional] = None) -> None:
        """
        Update the rates tensor based on the current state of the lattice.
        """
        if positions is None:
            self.initialize_rates()
            return
        n_orientations = len(Orientation)
        # compute a list of the neighbours of positions
        affected_cells = positions
        for index in range(len(positions)):
            affected_cells += self.lattice.get_neighbours(*positions[index])

        for x, y in affected_cells:
            self.rates[:n_orientations, y, x] = self.lattice.compute_local_tr(
                x, y, self.g
            )
            self.rates[n_orientations, y, x] = self.lattice.compute_local_tm(
                x, y, self.v0
            )

    def next_event_time(self) -> float:
        """
        Compute the time until the next event.

        :return: float - The time until the next event.
        """
        total_rate = self.rates.sum()
        assert (
            total_rate > 0
        ), "Total rate must be positive to sample from Exponential distribution."
        random_value = torch.rand(1).item()
        while random_value == 0:
            random_value = torch.rand(1).item()
        return -np.log(random_value) / total_rate

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
        rates_flat = self.rates.view(-1)
        total_rate = rates_flat.sum().item()

        if total_rate == 0:
            raise ValueError(
                "Cannot choose event: Total rate is zero, indicating no possible events."
            )

        # Generate a uniform random number between 0 and total_rate
        random_value = torch.rand(1, device=device) * total_rate

        # Use cumulative sum and binary search to find the event
        cumulative_rates = torch.cumsum(rates_flat, dim=0)
        chosen_index = torch.searchsorted(
            cumulative_rates,
            random_value.to(device),
        ).item()

        # Convert the flat index back into 3D index
        # convert the flat index back into 3D index using numpy.unravel_index because torch.unravel_index is not implemented yet

        # Temporary move to CPU for np.unravel_index, if necessary
        cpu_device = torch.device("cpu")
        if device != torch.device("cpu"):
            chosen_index_cpu = chosen_index
            event_type_index, y, x = np.unravel_index(
                chosen_index_cpu, self.rates.shape
            )
            # Convert results back to tensors and move to the original device
            event_type_index, y, x = (
                torch.tensor(event_type_index, device=device),
                torch.tensor(y, device=device),
                torch.tensor(x, device=device),
            )
            event_type_index = event_type_index.to(cpu_device).item()
            y = y.to(cpu_device).item()
            x = x.to(cpu_device).item()
        else:
            event_type_index, y, x = np.unravel_index(chosen_index, self.rates.shape)
            # No need to move back since we're on CPU

        event_type = EventType(event_type_index)

        return Event(event_type, x, y)

    def perform_event(self, event: Event) -> List[tuple]:
        """
        Execute the specified event on the lattice.

        This method determines the type of the given event (reorientation or migration)
        and performs the corresponding action on the lattice. In case of a reorientation
        event, it reorients the particle at the specified location. For a migration event,
        it moves the particle to a new location.

        :param event: The event object to be executed. It contains the event type and
                    the coordinates (x, y) on the lattice where the event occurs.
        """
        if self.with_stat:
            self.stat_event_counter += 1
        if event.is_reorientation():
            orientation = Orientation(event.etype.value)
            self.lattice.reorient_particle(event.x, event.y, orientation)
            return [(event.x, event.y)]
        elif event.is_migration():
            if self.with_stat:
                if self.lattice.orientation_map[event.y,event.x]==Orientation.RIGHT:
                    self.stat_flux_counter[event.y] += 1
                elif self.lattice.orientation_map[event.y,event.x]==Orientation.LEFT:
                    self.stat_flux_counter[event.y] -= 1
            new_pos = self.lattice.move_particle(event.x, event.y)
            return [(event.x, event.y)] + new_pos
        elif event.is_birth():
            new_pos = self.lattice.add_particle(event.x, event.y)
            return new_pos
        elif event.is_transport():
            direction = Orientation(event.etype.value - EventType.TRANSPORT_UP.value)
            new_pos = self.lattice.transport_particle(event.x, event.y, direction)
        else:
            raise ValueError(f"Unrecognized event type: {event.etype}")

    def run(self) -> Event:
        """
        Run the simulation for a single time step.

        :return: An Optional tuple (event_type, x, y) representing the event, or None.
        """
        # populate the lattice on the first iteration
        if self.lattice.particles.sum().item() == 0:
            self.populate_lattice(self.density)
        
        self.delta_t = self.next_event_time()
        self.t += self.delta_t
        event = self.choose_event()
        #JB: added to prevent bug "ValueError: Site (0, 0) is empty."
        if not event.y == 0:
            affected_sites = self.perform_event(event)
            # self.update_rates(affected_sites)
            self.update_rates()
        return event

    def apply_magnetic_field(self, direction: int = 0) -> None:
        """
        Apply the magnetic field to the lattice.
        """
        self.magnetic_field.set_direction(direction)
        self.magnetic_field.apply(self.lattice)
        self.update_rates()

    def get_magnetic_field_state(self) -> int:
        """
        Get the current direction of the magnetic field.

        :return: int - The current direction of the magnetic field.
        """
        return self.magnetic_field.get_current_direction()
    
    def init_stat(self) -> None:
        """
        Initialize statistical measurements
        """
        self.with_stat = True
        self.stat_counter = 0
        self.stat_mass = np.zeros(self.lattice.height,dtype=np.int32)
        self.stat_order_param_pi = np.zeros((self.lattice.height,2),dtype=np.int32)
        self.stat_order_param_psi = np.zeros(self.lattice.height,dtype=np.int32)
        self.stat_flux_counter = np.zeros(self.lattice.height,dtype=np.int32)
        self.stat_event_counter = 0

    def perform_stat(self) -> None:
        """
        Perform statistical measurements
        """
        if self.with_stat:
            self.stat_counter += 1
            count_neigh = np.array(self.lattice.occupancy_map).astype(int) * (  np.array(self.lattice.occupancy_map.roll(-1,0)).astype(int)
                                                                              + np.array(self.lattice.occupancy_map.roll(1,0)).astype(int)
                                                                              + np.array(self.lattice.occupancy_map.roll(-1,1)).astype(int)
                                                                              + np.array(self.lattice.occupancy_map.roll(1,1)).astype(int) )
            for iy in range(self.lattice.height):
                self.stat_mass[iy] += np.sum(np.array(self.lattice.occupancy_map[iy,:]).astype(int))
                self.stat_order_param_pi[iy,0] += (  np.sum(np.array(self.lattice.orientation_map[iy,:]==Orientation.RIGHT).astype(int))
                                                   - np.sum(np.array(self.lattice.orientation_map[iy,:]==Orientation.LEFT).astype(int)))
                self.stat_order_param_pi[iy,1] += (  np.sum(np.array(self.lattice.orientation_map[iy,:]==Orientation.UP).astype(int))
                                                   - np.sum(np.array(self.lattice.orientation_map[iy,:]==Orientation.DOWN).astype(int)))
                self.stat_order_param_psi[iy]  += np.sum(count_neigh[iy,:])
        else:
            raise ValueError(f"perform_stat() requires initializing statistics first")
            
    def dump_stat(self, fname) -> None:
        """
        Save statitistical measurements in ascii
        """
        with open(fname, "a") as file:
            pad = np.zeros((6,self.lattice.height),dtype=np.float32)
            pad[0,0] = self.t
            pad[0,1] = self.stat_counter
            pad[0,2] = self.stat_event_counter
            pad[1,:] = self.stat_mass
            pad[2,:] = self.stat_flux_counter
            pad[3,:] = self.stat_order_param_pi[:,0]
            pad[4,:] = self.stat_order_param_pi[:,1]
            pad[5,:] = self.stat_order_param_psi
            np.savetxt(file, pad, delimiter=' ', newline='\n')
        self.init_stat()
    
            
    
