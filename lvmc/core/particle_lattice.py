import torch
import torch.nn.functional as F
import numpy as np
import warnings
from enum import Enum
from typing import List, Tuple, Optional, Union


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Orientation(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


class ParticleLattice:
    """
    Class for the particle lattice.
    """

    ############################################
    ## Initialization and Basic Configuration ##
    ############################################

    NUM_ORIENTATIONS = len(Orientation)  # Class level constant

    def __init__(self, width: int, height: int, density: float = 0.0, mode="default"):
        """
        Initialize the particle lattice.
        :param width: Width of the lattice.
        :param height: Height of the lattice.
        :param density: Density of particles in the lattice
        """
        self.width = width
        self.height = height
        self.mode = mode

        # Initialize the paricles lattice as a 3D tensor with dimensions corresponding to
        # orientations, width, and height.
        self.particles = torch.zeros(
            (self.NUM_ORIENTATIONS, height, width), dtype=torch.bool, device=device
        )
        self.obstacles = torch.zeros((height, width), dtype=torch.bool, device=device)
        self.sinks = torch.zeros((height, width), dtype=torch.bool, device=device)
        # Initialize an array to store orientations of particles
        self.orientation_map = np.full(
            (height, width), None
        )  # None indicates no particle
        self.occupancy_map = torch.zeros(
            (height, width), dtype=torch.bool, device=device
        )

        # Particle tracking
        self.id_to_position = {}  # Dictionary to track particles
        self.position_to_particle_id = {}  # Dictionary to map positions to particle IDs
        self.next_particle_id = 0  # Counter to assign unique IDs to particles

        # Initialize the lattice with particles at a given density.
        self.populate(density)

        # Precompute deltas for each orientation
        self.orientation_deltas = {
            Orientation.UP: (0, -1),
            Orientation.DOWN: (0, 1),
            Orientation.LEFT: (-1, 0),
            Orientation.RIGHT: (1, 0),
        }

    def populate(self, density: float) -> int:
        """
        Initialize the lattice with particles at a given density.

        :param density: Density of particles to be initialized.
        :return: The number of particles added to the lattice.
        """
        num_cells = self.width * self.height
        num_particles = int(density * num_cells)

        # Randomly place particles
        positions = np.random.choice(num_cells, num_particles, replace=False)

        # Generate random orientations using the Orientation enum
        orientations = np.random.choice(list(Orientation), num_particles)

        n_added = 0

        for pos, ori in zip(positions, orientations):
            y, x = divmod(pos, self.width)  # Convert position to (x, y) coordinates
            if not self._is_obstacle(x, y) and self._is_empty(x, y):
                self.add_particle(x, y, ori)
                n_added += 1
        return n_added

    def get_params(self) -> dict:
        """
        Get the parameters of the lattice.
        :return: A dictionary of the lattice parameters.
        """
        return {
            "width": self.width,
            "height": self.height,
        }

    ####################################
    ## Validation and utility methods ##
    ####################################

    def _validate_coordinates(self, x: int, y: int):
        """
        Validate the coordinates to ensure they are within the lattice bounds.

        Parameters:
        x (int): x-coordinate to validate.
        y (int): y-coordinate to validate.

        Raises:
        ValueError: If the coordinates are outside the lattice bounds, specifying the invalid coordinates.
        """

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise IndexError(f"Coordinates ({x}, {y}) are out of lattice bounds.")

    def _create_index_to_symbol_mapping(self) -> dict:
        """
        Create a mapping between orientation indices and symbols for visualization.
        :return: A dictionary mapping orientation indices to symbols.
        """
        # Map each Orientation enum value to its corresponding symbol
        return {
            Orientation.UP.value: "↑",
            Orientation.DOWN.value: "↓",
            Orientation.LEFT.value: "←",
            Orientation.RIGHT.value: "→",
        }

    def _is_empty(self, x: int, y: int) -> bool:
        """
        Check if a cell is empty.

        :param x: x-coordinate of the lattice.
        :param y: y-coordinate of the lattice.
        :return: True if the no particle is present at the cell, False otherwise.
        """
        return not self.occupancy_map[y, x]

    def _get_target_position(self, x: int, y: int, orientation) -> tuple:
        """
        Get the expected position of a particle at (x, y) with a given orientation.

        :param orientation: The orientation of the particle.
        :param x: Current x-coordinate of the particle.
        :param y: Current y-coordinate of the particle.
        :return: The expected position of the particle.
        """

        # Validate coordinates
        self._validate_coordinates(x, y)

        # Validate occupancy
        self._validate_occupancy(x, y)

        # Calculate new position based on orientation
        delta_x, delta_y = self.orientation_deltas[orientation]
        new_x, new_y = (x + delta_x) % self.width, (y + delta_y) % self.height
        return new_x, new_y

    def _is_obstacle(self, x: int, y: int) -> bool:
        """
        Check if the specified cell is an obstacle.

        Parameters:
        x (int): x-coordinate of the cell in the lattice.
        y (int): y-coordinate of the cell in the lattice.

        Returns:
        bool: True if the cell is an obstacle, False otherwise.

        Raises:
        ValueError: If the coordinates are out of the lattice bounds.
        """
        self._validate_coordinates(x, y)
        return self.obstacles[y, x] == 1

    def _is_sink(self, x: int, y: int) -> bool:
        """
        Check if the specified cell is a sink.

        Parameters:
        x (int): x-coordinate of the cell in the lattice.
        y (int): y-coordinate of the cell in the lattice.

        Returns:
        bool: True if the cell is a sink, False otherwise.

        Raises:
        IndexError: If the coordinates are out of the lattice bounds.
        """
        self._validate_coordinates(x, y)
        return self.sinks[y, x] == 1

    def _validate_availability(self, x: int, y: int) -> None:
        """
        Validate that the specified cell is available for particle movement.

        Parameters:
        x (int): x-coordinate of the cell in the lattice.
        y (int): y-coordinate of the cell in the lattice.

        Raises:
        IndexError: If the coordinates are out of the lattice bounds.
        ValueError: If the specified cell is an obstacle.
        ValueError: If the specified cell is a non-empty.
        """
        if self.mode == "optimized":
            return
        self._validate_coordinates(x, y)
        if self._is_obstacle(x, y):
            raise ValueError(f"Site ({x}, {y}) is an obstacle.")
        if not self._is_empty(x, y):
            raise ValueError(f"Site ({x}, {y}) is not empty.")

    def _validate_occupancy(self, x: int, y: int) -> None:
        """
        Validate that the specified cell is occupied by a particle.

        Parameters:
        x (int): x-coordinate of the cell in the lattice.
        y (int): y-coordinate of the cell in the lattice.

        Raises:
        IndexError: If the coordinates are out of the lattice bounds.
        ValueError: If the specified cell is empty.
        """
        if self.mode == "optimized":
            return
        self._validate_coordinates(x, y)
        if self._is_empty(x, y):
            raise ValueError(f"Site ({x}, {y}) is empty.")

    def _update_tracking(self, id: int, x: int, y: int) -> None:
        """
        Update the particle tracking dictionaries.
        :param id: The id of the particle.
        :param x: The x-coordinate of the particle.
        """
        self.id_to_position[id] = (x, y)
        self.position_to_particle_id[(x, y)] = id

    @property
    def shape(self):
        """
        Returns the shape (height, width) of the lattice.
        """
        return (self.height, self.width)

    @property
    def density(self):
        """
        Returns the density of the lattice, calculated as the ratio of occupied cells to total cells.
        """
        num_occupied_cells = (
            self.particles.any(dim=0).sum().item()
        )  # Count occupied cells
        total_cells = self.width * self.height
        return num_occupied_cells / total_cells if total_cells > 0 else 0

    @property
    def n_particles(self):
        return self.particles.sum().item()

    @property
    def is_empty(self):
        return torch.all(~self.occupancy_map)

    ###################################
    ## Particle Manipulation Methods ##
    ###################################

    def add_particle(self, x: int, y: int, orientation: Orientation = None) -> None:
        """
        Add a particle with a specific orientation at (x, y).

        :param x: x-coordinate of the lattice.
        :param y: y-coordinate of the lattice.
        :param orientation: Orientation of the particle, as an instance of the Orientation enum.
        """
        # Validate that orientation is an instance of Orientation
        if orientation is None:
            orientation = np.random.choice(list(Orientation))

        if not isinstance(orientation, Orientation):
            raise ValueError("orientation must be an instance of Orientation enum.")

        # Validate that the specified cell is available for particle movement
        self._validate_availability(x, y)

        self.particles[orientation.value, y, x] = True  # Add particle to the lattice
        self.orientation_map[y, x] = orientation
        self.occupancy_map[y, x] = True

        self._update_tracking(
            self.next_particle_id, x, y
        )  # update the particle tracking dictionaries

        self.next_particle_id += 1  # increment the next particle id

    def remove_particle(self, x: int, y: int) -> None:
        """
        Remove a particle from a specific node in the lattice.

        :param x: x-coordinate of the node.
        :param y: y-coordinate of the node.
        """
        # Validate coordinates
        self._validate_coordinates(x, y)
        self.particles[self.orientation_map[y, x].value, y, x] = False
        self.orientation_map[y, x] = None
        self.occupancy_map[y, x] = False

    def get_particle_orientation(self, x: int, y: int) -> Orientation:
        """
        Get the orientation of a particle at (x, y).

        :param x: x-coordinate of the particle.
        :param y: y-coordinate of the particle.
        :return: The orientation of the particle as an Orientation enum instance. None if no particle is found.
        """
        self._validate_occupancy(x, y)
        return self.orientation_map[y, x]

    def move_particle(self, x: int, y: int) -> List[tuple]:
        """
        Move a particle at (x, y) with a given orientation to the new position determined by its current orientation.
        :param x: Current x-coordinate of the particle.
        :param y: Current y-coordinate of the particle.
        :return: A list of tuples representing the new position of the particle.
        :raises ValueError: If no particle is found at the given location.
        """

        self._validate_occupancy(
            x, y
        )  # Check if the particle exists at the given location
        orientation = self.get_particle_orientation(x, y)
        # Get the expected position of the particle
        new_x, new_y = self._get_target_position(x, y, orientation)

        particle_id = self.position_to_particle_id.pop((x, y))
        self._update_tracking(particle_id, new_x, new_y)

        if self._is_sink(new_x, new_y):
            self.remove_particle(x, y)
            return []

        # Directly update the particle's position in the lattice
        self.particles[orientation.value, y, x] = False
        self.particles[orientation.value, new_y, new_x] = True

        # Update the orientation map
        self.orientation_map[y, x] = None
        self.orientation_map[new_y, new_x] = orientation
        # Update the occupancy map
        self.occupancy_map[y, x] = False
        self.occupancy_map[new_y, new_x] = True

        return [(new_x, new_y)]

    def transport_particle(self, x: int, y: int, direction: Orientation) -> List[tuple]:
        """
        Move a particle at (x, y) with a given orientation to the new position determined by the prescribed orientation.
        :param x: Current x-coordinate of the particle.
        :param y: Current y-coordinate of the particle.
        :return: A list of tuples representing the new position of the particle.
        :raises ValueError: If no particle is found at the given location.
        """

        self._validate_occupancy(
            x, y
        )  # Check if the particle exists at the given location

        # Get the expected position of the particle, and the old orientation
        new_x, new_y = self._get_target_position(x, y, direction)
        orientation = self.get_particle_orientation(x, y)

        # get the id of the particle at (x, y)
        particle_id = self.position_to_particle_id.pop((x, y), None)
        self._update_tracking(particle_id, new_x, new_y)
        if self._is_sink(new_x, new_y):
            self.remove_particle(x, y)
            return []

        self.remove_particle(x, y)
        self.add_particle(new_x, new_y, orientation)
        return [(new_x, new_y)]

    def reorient_particle(self, x: int, y: int, new_orientation: Orientation) -> bool:
        """
        Reorient a particle at (x, y) to a new orientation.

        :param x: x-coordinate of the particle.
        :param y: y-coordinate of the particle.
        :param new_orientation: The new orientation for the particle as an Orientation enum instance.
        :return: True if the particle was reoriented successfully, False otherwise.
        """
        # Validate that new_orientation is an instance of Orientation
        if not isinstance(new_orientation, Orientation):
            raise ValueError(
                f"{new_orientation=} must be an instance of Orientation enum."
            )

        current_orientation = self.get_particle_orientation(x, y)

        # Update the orientation in the particles tensor and orientation_map
        self.particles[current_orientation.value, y, x] = False
        self.particles[new_orientation.value, y, x] = True
        self.orientation_map[y, x] = new_orientation

    ##################################
    ## Obstacle and sink management ##
    ##################################

    def set_obstacle(self, x: int, y: int) -> None:
        """
        Set an obstacle at the specified position in the lattice,
        provided the cell is empty and not already an obstacle or a sink.

        Parameters:
        x (int): The x-coordinate of the position.
        y (int): The y-coordinate of the position.

        Raises:
        ValueError: If the specified position is outside the lattice bounds or already occupied.
        """
        self._validate_coordinates(x, y)
        self._validate_availability(x, y)

        self.obstacles[y, x] = True

    def set_sink(self, x: int, y: int) -> None:
        """
        Set a sink at the specified position in the lattice,
        provided the cell is empty and not already an obstacle or a sink.

        Parameters:
        x (int): The x-coordinate of the position.
        y (int): The y-coordinate of the position.

        Raises:
        ValueError: If the specified position is outside the lattice bounds or already occupied.
        """
        self._validate_coordinates(x, y)
        self._validate_availability(x, y)
        self.sinks[y, x] = True

    def set_obstacles(self, obstacles: torch.Tensor) -> None:
        """
        Set the obstacles for the lattice.
        :param obstacles: A binary matrix indicating the obstacle cells.
        """
        if obstacles.shape != (self.height, self.width):
            raise ValueError(
                f"Obstacles tensor must match the lattice dimensions. \n >>> {obstacles.shape=}, {(self.height, self.width)=}"
            )
        self.obstacles = obstacles

    def set_sinks(self, sinks: torch.Tensor) -> None:
        """
        Set the sinks for the lattice.
        :param sinks: A binary matrix indicating the sink cells.
        """
        if sinks.shape != (self.height, self.width):
            raise ValueError("Sinks tensor must match the lattice dimensions.")
        self.sinks = sinks

    ##################################
    ## Transition Rates Computation ##
    ##################################

    def compute_tm(self, v0: float = 1.0) -> None:
        """
        Compute the migration transition rate tensor TM with periodic boundary conditions.
        """
        # Calculate empty cells (where no particle nor obstacle is present)
        empty_cells = ~(self.particles.sum(dim=0) + self.obstacles).bool()

        # Calculate potential moves in each direction
        TM_up = self.particles[Orientation.UP.value] * empty_cells.roll(
            shifts=1, dims=0
        )
        TM_down = self.particles[Orientation.DOWN.value] * empty_cells.roll(
            shifts=-1, dims=0
        )
        TM_left = self.particles[Orientation.LEFT.value] * empty_cells.roll(
            shifts=1, dims=1
        )
        TM_right = self.particles[Orientation.RIGHT.value] * empty_cells.roll(
            shifts=-1, dims=1
        )

        # Combine all moves
        TM = TM_up + TM_down + TM_left + TM_right

        return TM * v0

    def compute_log_tr(self) -> torch.Tensor:
        """
        Compute the reorientation transition log rate tensor.

        :return: The reorientation transition log rate tensor.
        """

        # Common kernel for convolution
        kernel = (
            torch.tensor(
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=device
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        padded_particles = F.pad(self.particles, pad=(1, 1, 1, 1), mode="circular")
        # Convolve each orientation layer of the lattice
        log_TR_tensor = torch.zeros(
            (ParticleLattice.NUM_ORIENTATIONS, self.height, self.width),
            dtype=torch.float32,
            device=device,
        )
        for orientation in Orientation:
            input_tensor = (
                padded_particles[orientation.value].unsqueeze(0).unsqueeze(0).float()
            )
            log_TR_tensor[orientation.value] = F.conv2d(
                input_tensor, kernel, padding=0
            )[0, 0]

        # Adjusting the log_TR tensor based on orientation vectors
        log_TR_tensor[Orientation.UP.value], log_TR_tensor[Orientation.DOWN.value] = (
            log_TR_tensor[Orientation.UP.value] - log_TR_tensor[Orientation.DOWN.value],
            log_TR_tensor[Orientation.DOWN.value] - log_TR_tensor[Orientation.UP.value],
        )
        (
            log_TR_tensor[Orientation.LEFT.value],
            log_TR_tensor[Orientation.RIGHT.value],
        ) = (
            log_TR_tensor[Orientation.LEFT.value]
            - log_TR_tensor[Orientation.RIGHT.value],
            log_TR_tensor[Orientation.RIGHT.value]
            - log_TR_tensor[Orientation.LEFT.value],
        )

        return log_TR_tensor

    def compute_log_tr_obstacles(self) -> torch.Tensor:
        """
        Compute the reorientation transition log rate tensor near obstacles.
        """
        log_tr_obstacles = torch.zeros(
            (ParticleLattice.NUM_ORIENTATIONS, self.height, self.width),
            dtype=torch.float32,
            device=device,
        )

        obstacle_interaction_strength = 0.5

        # Mapping of orientations to their perpendicular particle orientations
        perpendicular_orientations = {
            Orientation.LEFT: (Orientation.UP, Orientation.DOWN),
            Orientation.RIGHT: (Orientation.UP, Orientation.DOWN),
            Orientation.UP: (Orientation.RIGHT, Orientation.LEFT),
            Orientation.DOWN: (Orientation.RIGHT, Orientation.LEFT),
        }

        for orientation, (perp1, perp2) in perpendicular_orientations.items():
            for shift in [-1, 1]:
                # Choose the dimension perpendicular to particle orientation for obstacle shifting
                # 0 for horizontal (LEFT, RIGHT) orientations (shift vertically),
                # 1 for vertical (UP, DOWN) orientations (shift horizontally).
                dim = 0 if orientation in [Orientation.LEFT, Orientation.RIGHT] else 1
                log_tr_obstacles[orientation.value] += (
                    (self.particles[perp1.value] + self.particles[perp2.value])
                    * self.obstacles.roll(shifts=shift, dims=dim)
                    * obstacle_interaction_strength
                )

        return log_tr_obstacles

    def compute_tr(self, g: float = 1.0) -> None:
        """
        Compute the reorientation transition rate tensor TR.

        :param g: Parameter controlling alignment sensitivity. Default is 1.0.
        :type g: float
        """
        # Calculate occupied cells (where at least one particle is present)
        occupied_cells = self.particles.sum(dim=0).bool()

        log_tr = self.compute_log_tr() + self.compute_log_tr_obstacles()
        tr = torch.exp(g * log_tr) * occupied_cells

        return tr * (torch.ones_like(self.particles) ^ self.particles)

    def compute_local_tm(self, x: int, y: int, v0: float = 1.0) -> float:
        """
        Compute the local migration transition rate at (x, y).
        It's v0 if the target cell is empty, 0 otherwise.
        """
        # Validate coordinates
        self._validate_coordinates(x, y)

        if self._is_empty(x, y):
            return 0.0
        # Get the coordinates of the target cell
        orientation = self.get_particle_orientation(x, y)
        new_x, new_y = self._get_target_position(x, y, orientation)

        # Check if the target cell is empty
        if self._is_empty(new_x, new_y):
            return v0
        else:
            return 0.0

    def compute_local_log_tr(self, x: int, y: int) -> torch.Tensor:
        pass

    def compute_local_tr(self, x: int, y: int, g: float = 1.0) -> torch.Tensor:
        """
        Compute the local reorientation transition rate at (x, y).
        If no particle is present at (x, y), return zero rates.
        """
        # Check if the cell is empty
        if torch.sum(self.particles[:, y, x]) == 0:
            return torch.zeros((ParticleLattice.NUM_ORIENTATIONS), dtype=torch.float32)

        local_log_tr = self.compute_local_log_tr(x, y)
        local_tr = torch.exp(g * local_log_tr)

        return local_tr

    def get_neighbours(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get the coordinates of the four neighbouring cells (up, down, left, right)
        with periodic boundary conditions.

        :param x: The x-coordinate of the cell.
        :param y: The y-coordinate of the cell.

        :return: A list of tuples, each representing the coordinates of a neighbor.
        """
        return [
            ((x - 1) % self.width, y),  # Up
            ((x + 1) % self.width, y),  # Down
            (x, (y - 1) % self.height),  # Left
            (x, (y + 1) % self.height),  # Right
        ]

    ##################################
    ## State Querying and Reporting ##
    ##################################

    def query_lattice_state(self) -> torch.Tensor:
        """
        Query the current state of the lattice.

        :return: The state of the lattice.
        """
        return self.particles

    ###################################
    ## String Representation Methods ##
    ###################################

    def __str__(self) -> str:
        """
        String representation of the lattice.
        :return: A string representation of the lattice.
        """
        index_to_symbol = self._create_index_to_symbol_mapping()
        obstacle_symbol = "■"  # Symbol for obstacles
        sink_symbol = "▼"  # Symbol for sinks
        particle_in_sink_symbol = "✱"  # Symbol for particle in a sink
        lattice_str = ""

        for y in range(self.height):
            row_str = ""
            for x in range(self.width):
                if self.obstacles[y, x]:
                    row_str += obstacle_symbol
                elif self.sinks[y, x]:
                    if not self._is_empty(x, y):  # Check for particle in sink
                        row_str += particle_in_sink_symbol
                    else:
                        row_str += sink_symbol
                elif self._is_empty(x, y):
                    row_str += "·"  # Use a dot for empty cells
                else:
                    orientation_index = self.get_particle_orientation(x, y).value
                    symbol = index_to_symbol[orientation_index]
                    row_str += symbol
                row_str += " "  # Add space between cells
            lattice_str += row_str
            if y < self.height - 1:
                lattice_str += (
                    "\n"  # Add a newline character after each row except the last
                )

        return lattice_str

    def __repr__(self) -> str:
        """
        Representation of the lattice.
        :return: A string representation of the lattice.
        """
        return self.__str__()

    def __getitem__(self, key):
        """
        Enables slicing of the ParticleLattice and validates the slicing keys.

        :param key: The slicing key.
        :return: A new ParticleLattice instance with the sliced data.
        """
        # Validate and adjust the slicing keys
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Slicing key must be a tuple of two slice objects.")

        key_y, key_x = key
        key_y = slice(
            max(0, key_y.start or 0),
            min(self.height, key_y.stop or self.height),
            key_y.step,
        )
        key_x = slice(
            max(0, key_x.start or 0),
            min(self.width, key_x.stop or self.width),
            key_x.step,
        )
        adjusted_key = (key_y, key_x)

        # Create a new instance of ParticleLattice
        new_lattice = ParticleLattice(self.width, self.height, mode=self.mode)

        # Slicing the tensors
        new_lattice.particles = self.particles[:, adjusted_key[0], adjusted_key[1]]
        new_lattice.obstacles = self.obstacles[adjusted_key[0], adjusted_key[1]]
        new_lattice.sinks = self.sinks[adjusted_key[0], adjusted_key[1]]

        # Update the width and height of the new lattice
        new_lattice.width = new_lattice.obstacles.shape[1]
        new_lattice.height = new_lattice.obstacles.shape[0]

        # Resetting other attributes
        new_lattice.id_to_position = {}
        new_lattice.position_to_particle_id = {}
        new_lattice.next_particle_id = 0

        return new_lattice
