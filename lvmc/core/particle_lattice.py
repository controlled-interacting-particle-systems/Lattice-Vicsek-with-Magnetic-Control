import torch
import torch.nn.functional as F
import numpy as np
import warnings
from enum import Enum


class Orientation(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


class ParticleLattice:
    """
    Class for the particle lattice.
    """

    NUM_ORIENTATIONS = len(Orientation)  # Class level constant

    def __init__(
        self,
        width: int,
        height: int,
        density: float = 0.0,
    ):
        """
        Initialize the particle lattice.
        :param width: Width of the lattice.
        :param height: Height of the lattice.
        :param density: Density of particles in the lattice
        """
        self.width = width
        self.height = height

        # Initialize the paricles lattice as a 3D tensor with dimensions corresponding to
        # orientations, width, and height.
        self.particles = torch.zeros(
            (self.NUM_ORIENTATIONS, height, width), dtype=torch.bool
        )
        self.obstacles = torch.zeros((height, width), dtype=torch.bool)
        self.sinks = torch.zeros((height, width), dtype=torch.bool)

        # Initialize the lattice with particles at a given density.
        self._initialize_lattice(density)

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
            raise ValueError(f"Coordinates ({x}, {y}) are out of lattice bounds.")

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
            lattice_str += row_str + "\n"

        return lattice_str

    def __repr__(self) -> str:
        """
        Representation of the lattice.
        :return: A string representation of the lattice.
        """
        return self.__str__()

    def _initialize_lattice(self, density: float) -> None:
        """
        Initialize the lattice with particles at a given density.

        :param density: Density of particles to be initialized.
        """
        num_cells = self.width * self.height
        num_particles = int(density * num_cells)

        # Randomly place particles
        positions = np.random.choice(num_cells, num_particles, replace=False)

        # Generate random orientations using the Orientation enum
        orientations = np.random.choice(list(Orientation), num_particles)

        for pos, ori in zip(positions, orientations):
            y, x = divmod(pos, self.width)  # Convert position to (x, y) coordinates
            if not self._is_obstacle(x, y):
                self.add_particle(x, y, ori)

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
        if self.obstacles is None:
            self.obstacles = torch.zeros((self.height, self.width), dtype=torch.bool)
        if 0 <= x < self.width and 0 <= y < self.height:
            if not self._is_empty(x, y) or self._is_sink(x, y):
                raise ValueError(
                    "Cannot place an obstacle on a non-empty cell or a cell with a sink."
                )
            if self._is_sink(x, y):
                warnings.warn(
                    "Placing an obstacle on a sink will remove the sink. Make sure that this is intended",
                    stacklevel=2,
                )
                self.sinks[y, x] = False
            if self._is_obstacle(x, y):
                warnings.warn(
                    "Trying to place an obstacle on a cell that is already an obstacle. Please make sure that this is intended.",
                    stacklevel=2,
                )
            self.obstacles[y, x] = True
        else:
            raise ValueError("Position is outside the lattice bounds.")

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
        if self.sinks is None:
            self.sinks = torch.zeros((self.height, self.width), dtype=torch.bool)
        if 0 <= x < self.width and 0 <= y < self.height:
            if not self._is_empty(x, y) or self._is_obstacle(x, y):
                raise ValueError(
                    "Cannot place a sink on a non-empty cell or a cell with an obstacle."
                )
            if self._is_sink(x, y):
                warnings.warn(
                    "Trying to place a sink on a cell that is already a sink. Please make sure that this is intended.",
                    stacklevel=2,
                )
            self.sinks[y, x] = True
        else:
            raise ValueError("Position is outside the lattice bounds.")

    def _is_empty(self, x: int, y: int) -> bool:
        """
        Check if a cell is empty.

        :param x: x-coordinate of the lattice.
        :param y: y-coordinate of the lattice.
        :return: True if the no particle is present at the cell, False otherwise.
        """
        return not self.particles[:, y, x].any()

    def add_particle(self, x: int, y: int, orientation: Orientation) -> None:
        """
        Add a particle with a specific orientation at (x, y).

        :param x: x-coordinate of the lattice.
        :param y: y-coordinate of the lattice.
        :param orientation: Orientation of the particle, as an instance of the Orientation enum.
        """
        # Validate that orientation is an instance of Orientation
        if not isinstance(orientation, Orientation):
            raise ValueError("orientation must be an instance of Orientation enum.")

        if self._is_empty(x, y) and not self._is_obstacle(x, y):
            self.particles[orientation.value, y, x] = True
        else:
            raise ValueError(
                f"Cannot add particle, cell ({x},{y}) is occupied or is an obstacle."
            )

    def remove_particle(self, x: int, y: int) -> None:
        """
        Remove a particle from a specific node in the lattice.

        :param x: x-coordinate of the node.
        :param y: y-coordinate of the node.
        """
        if self._is_empty(x, y):
            warnings.warn(
                "Trying to remove a particle from an empty cell. Please make sure that this is intended.",
                stacklevel=2,
            )
        if self._is_obstacle(x, y):
            warnings.warn(
                "Trying to remove a particle from an obstacle. Please make sure that this is intended.",
                stacklevel=2,
            )
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise ValueError(
                "Cannot remove particle, cell is outside the lattice bounds."
            )
        self.particles[:, y, x] = False  # Remove particle from all orientations

    def query_lattice_state(self) -> torch.Tensor:
        """
        Query the current state of the lattice.

        :return: The state of the lattice.
        """
        return self.particles

    def compute_tm(self, v0: float = 1.0) -> None:
        """
        Compute the migration transition rate tensor TM with periodic boundary conditions.
        """
        # Calculate empty cells (where no particle is present)
        empty_cells = ~self.particles.sum(dim=0).bool()

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
            torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Convolve each orientation layer of the lattice
        log_TR_tensor = torch.zeros(
            (ParticleLattice.NUM_ORIENTATIONS, self.height, self.width),
            dtype=torch.float32,
        )
        for orientation in Orientation:
            input_tensor = (
                self.particles[orientation.value].unsqueeze(0).unsqueeze(0).float()
            )
            log_TR_tensor[orientation.value] = F.conv2d(
                input_tensor, kernel, padding=1
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

    def compute_tr(self, g: float = 1.0) -> None:
        """
        Compute the reorientation transition rate tensor TR.

        :param g: Parameter controlling alignment sensitivity. Default is 1.0.
        :type g: float
        """
        # Calculate occupied cells (where at least one particle is present)
        occupied_cells = self.particles.sum(dim=0).bool()

        log_tr = self.compute_log_tr()
        tr = torch.exp(g * log_tr) * occupied_cells

        return tr * (torch.ones_like(self.particles) ^ self.particles)

    def _get_target_position(self, x: int, y: int, orientation: Orientation) -> tuple:
        """
        Get the expected position of a particle at (x, y) with a given orientation.

        :param x: Current x-coordinate of the particle.
        :param y: Current y-coordinate of the particle.
        :param orientation: Current orientation of the particle which determines the direction of movement.
        :return: The expected position of the particle.
        :raises ValueError: If invalid orientation is provided.
        """
        # Validate the orientation
        if not isinstance(orientation, Orientation):
            raise ValueError("Invalid orientation type.")

        # Calculate new position based on orientation
        if orientation == Orientation.UP:
            new_x, new_y = x, (y - 1) % self.height
        elif orientation == Orientation.DOWN:
            new_x, new_y = x, (y + 1) % self.height
        elif orientation == Orientation.LEFT:
            new_x, new_y = (x - 1) % self.width, y
        elif orientation == Orientation.RIGHT:
            new_x, new_y = (x + 1) % self.width, y
        else:
            raise ValueError("Invalid orientation index.")

        return (new_x, new_y)

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
        ValueError: If the coordinates are out of the lattice bounds.
        """
        self._validate_coordinates(x, y)
        return self.sinks[y, x] == 1

    def get_particle_orientation(self, x: int, y: int) -> Orientation:
        """
        Get the orientation of a particle at (x, y).

        :param x: x-coordinate of the particle.
        :param y: y-coordinate of the particle.
        :return: The orientation of the particle as an Orientation enum instance.
        :raises ValueError: If no particle is found at the given location.
        """
        # If no particle is found at the given location, raise a value error
        if self._is_empty(x, y):
            raise ValueError(f"No particle found at the given location: ({x}, {y}).")

        # Get the orientation of the particle
        orientation_index = self.particles[:, y, x].nonzero(as_tuple=True)[0].item()

        # Convert the index to an Orientation enum
        try:
            return Orientation(orientation_index)
        except ValueError:
            raise ValueError(f"Invalid orientation index found: {orientation_index}")

    def move_particle(self, x: int, y: int) -> bool:
        """
        Move a particle at (x, y) with a given orientation to the new position determined by its current orientation.
        :param x: Current x-coordinate of the particle.
        :param y: Current y-coordinate of the particle.
        :return: True if the particle was moved successfully, False otherwise.
        :raises ValueError: If no particle is found at the given location.
        """
        # Check if the particle exists at the given location
        if self._is_empty(x, y):
            raise ValueError("No particle found at the given location.")

        # Get the current orientation of the particle at (x, y)
        orientation = self.get_particle_orientation(x, y)

        # Get the expected position of the particle
        new_x, new_y = self._get_target_position(x, y, orientation)

        # Check if the new position is occupied or is an obstacle
        if self._is_obstacle(new_x, new_y) or not self._is_empty(new_x, new_y):
            warnings.warn(
                "Cannot move particle to the target position as there is an obstacle or another particle there.",
                stacklevel=2,
            )
            return False

        # Check if the new position is a sink, if so remove the particle
        if self._is_sink(new_x, new_y):
            self.remove_particle(x, y)
            return True

        # Move the particle
        self.remove_particle(x, y)
        self.add_particle(new_x, new_y, orientation)

        return True

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

        # Get the current orientation of the particle at (x, y)
        current_orientation = self.get_particle_orientation(x, y)

        # If the new orientation is the same as the current one, return False
        if current_orientation == new_orientation:
            return False

        # Reorient the particle
        self.remove_particle(x, y)
        self.add_particle(x, y, new_orientation)

        return True

    def get_params(self) -> dict:
        """
        Get the parameters of the lattice.
        :return: A dictionary of the lattice parameters.
        """
        return {
            "width": self.width,
            "height": self.height,
        }

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
