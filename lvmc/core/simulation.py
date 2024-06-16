import torch
import numpy as np
from typing import Optional, Tuple
from lvmc.core.particle_lattice import ParticleLattice, Orientation
from lvmc.core.magnetic_field import MagneticField
from lvmc.core.flow import PoiseuilleFlow
from enum import Enum, auto
from typing import NamedTuple, List, Optional
from typing import Tuple


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
        self,
        g: float,
        v0: float,
        seed: Optional[int] = 1337,
    ) -> None:
        """
        Initialize the simulation with base parameters.

        :param g: Alignment sensitivity parameter.
        :param v0: Base transition rate for particle movement.
        """

        self.g = g
        self.v0 = v0
        self.n_event_types = 6  # base (1 migration and 4 reorientations + 1 birth)
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)
        # Initialize time
        self.t = 0.0

    def add_lattice(self, width: int, height: int) -> None:
        """
        Add a lattice to the simulation.

        :param width: The width of the lattice.
        :param height: The height of the lattice.
        """
        self.width = width
        self.height = height
        self.lattice = ParticleLattice(width, height, generator=self.generator)
        return self

    def add_control_field(self, direction: int = 0) -> None:
        """
        Add a control field to the simulation.

        :param direction: The direction of the control field.
        """
        self.control_field = MagneticField(direction)
        return self

    def add_flow(self, flow_params: dict) -> None:
        """
        Add a flow to the simulation.

        :param flow_params: A dictionary containing the parameters of the flow.
        """
        if flow_params["type"] == "Poiseuille":
            self.flow = PoiseuilleFlow(
                width=self.width, height=self.height, v1=flow_params["v1"]
            )
            print("Added Poiseuille flow to the simulation.")
        # Increment the number of event types
        self.n_event_types += 4
        return self

    def add_obstacles(self, obstacles: torch.Tensor) -> None:
        """
        Add obstacles to the lattice.

        :param obstacles: A tensor representing the obstacles on the lattice.
        """
        # Check the shape of the obstacles tensor
        assert obstacles.shape == (
            self.height,
            self.width,
        ), "The shape of the obstacles tensor is not correct."
        self.lattice.set_obstacles(obstacles)
        return self

    def add_sinks(self, sinks: torch.Tensor) -> None:
        """
        Add sinks to the lattice.

        :param sinks: A tensor representing the sinks on the lattice.
        """
        # Check the shape of the sinks tensor
        assert sinks.shape == (
            self.height,
            self.width,
        ), "The shape of the sinks tensor is not correct."
        self.lattice.set_sinks(sinks)
        return self

    def add_sources(self, sources: torch.Tensor) -> None:
        """
        Add sources to the lattice.

        :param sources: A tensor representing the sources on the lattice.
        """
        # Check the shape of the sources tensor
        assert sources.shape == (
            self.height,
            self.width,
        ), "The shape of the sources tensor is not correct."
        self.lattice.set_sources(sources)
        return self

    def add_particles(self, density: float) -> None:
        """
        Populate the lattice with particles.

        :param density: The density of the particles.
        """
        n_added = self.lattice.populate(density)
        print(f"Added {n_added} particles to the lattice.")
        return self

    def add_particles_from_list(self, part_up, part_left, part_down, part_right) -> None:
        """
        Populate the lattice with particles from lists.

        :param part_up: List of particle positions with an up orientation.
        :param part_left: List of particle positions with a left orientation.
        :param part_down: List of particle positions with a down orientation.
        :param part_right: List of particle positions with a right orientation.
        """
        ntot = 0
        if len(part_up) == 2:
            if any(len(row) != len(part_up[0]) for row in part_up):
                raise ValueError("Input part_up must be a table with 2 rows and equal number of columns in each row")
            n_up = len(part_up[0])
            for i in range(n_up):
                self.lattice.add_particle(part_up[0][i],part_up[1][i],Orientation.UP)
            ntot += n_up
        if len(part_left) == 2:
            if any(len(row) != len(part_left[0]) for row in part_left):
                raise ValueError("Input part_left must be a table with 2 rows and equal number of columns in each row")
            n_left = len(part_left[0])
            for i in range(n_left):
                self.lattice.add_particle(part_left[0][i],part_left[1][i],Orientation.LEFT)
            ntot += n_left
        if len(part_down) == 2:
            if any(len(row) != len(part_down[0]) for row in part_down):
                raise ValueError("Input part_down must be a table with 2 rows and equal number of columns in each row")
            n_down = len(part_down[0])
            for i in range(n_down):
                self.lattice.add_particle(part_down[0][i],part_down[1][i],Orientation.DOWN)
            ntot += n_down
        if len(part_right) == 2:
            if any(len(row) != len(part_right[0]) for row in part_right):
                raise ValueError("Input part_right must be a table with 2 rows and equal number of columns in each row")
            n_right = len(part_right[0])
            for i in range(n_right):
                self.lattice.add_particle(part_right[0][i],part_right[1][i],Orientation.RIGHT)
            ntot += n_right
        
        print(f"Added {ntot} particles to the lattice.")
        return self

    def build(self) -> None:
        """
        Build the simulation.
        """
        self.initialize_rates()
        self.compute_rates()
        return self

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
        self.rates = torch.zeros(
            self.n_event_types, self.height, self.width, device=device
        )

    def compute_rates(self) -> None:
        """
        Compute the rates of the different events on the lattice.
        """
        ind_offset = 0
        n_orientations = len(Orientation)
        self.rates[ind_offset:n_orientations] = self.lattice.compute_tr(self.g)
        self.rates[EventType.MIGRATION.value] = self.lattice.compute_tm(self.v0)
        ind_offset += n_orientations + 1
        try:
            # check that sources is an attribute of the lattice
            assert hasattr(self.lattice, "sources")
            self.rates[EventType.BIRTH.value] = self.lattice.compute_birth_rates(
                self.v0
            )
        except AssertionError:
            # print a warning if the sources attribute is not found but still assign the birth rate to the rates tensor
            self.rates[EventType.BIRTH.value] = 0
        ind_offset += 1

        try:
            # check that flow is an attribute of the simulation
            assert hasattr(self, "flow")
            self.rates[ind_offset : ind_offset + n_orientations] = self.flow.compute_tm(
                self.lattice.occupancy_map
            )
            self.rates[:n_orientations] += self.flow.compute_tr(self.lattice)

        except AssertionError:
            pass

    def update_rates(self, positions: list[Optional[int]] = None) -> None:
        """
        Update the rates tensor based on the current state of the lattice.
        """
        if positions is None:
            self.compute_rates()
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
        random_value = 0
        while random_value == 0:
            random_value = torch.rand(1, generator=self.generator).item()
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
        random_value = (
            torch.rand(1, device=device, generator=self.generator) * total_rate
        )

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
        if event.is_reorientation():
            orientation = Orientation(event.etype.value)
            self.lattice.reorient_particle(event.x, event.y, orientation)
            return [(event.x, event.y)]
        elif event.is_migration():
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

        self.delta_t = self.next_event_time()
        self.t += self.delta_t
        event = self.choose_event()
        affected_sites = self.perform_event(event)
        # self.update_rates(affected_sites)
        self.update_rates()
        return event

    def apply_magnetic_field(self, direction: int = 0) -> None:
        """
        Apply the magnetic field to the lattice.
        """
        self.control_field.set_direction(direction)
        self.control_field.apply(self.lattice)
        self.update_rates()

    def get_magnetic_field_state(self) -> int:
        """
        Get the current direction of the magnetic field.

        :return: int - The current direction of the magnetic field.
        """
        return self.control_field.get_current_direction()
