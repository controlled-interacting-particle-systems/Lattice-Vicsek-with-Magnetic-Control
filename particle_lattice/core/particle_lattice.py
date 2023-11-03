import torch
import torch.nn.functional as F
import numpy as np


class ParticleLattice:
    """
    Class for the particle lattice.
    """

    NUM_ORIENTATIONS = 4  # Class level constant, shared by all instances of the class. Number of possible orientations for a particle
    def __init__(self, width, height, num_layers=None):
        """
        Initialize the particle lattice.

        :param width: Width of the lattice.
        :type width: int
        :param height: Height of the lattice.
        :type height: int
        :param num_layers: Number of possible orientations (4) for a particle + possibly other cell types.
        :type num_layers: int
        """
        if num_layers is None:
            num_layers = ParticleLattice.NUM_ORIENTATIONS
        self.width = width
        self.height = height
        self.num_layers = num_layers
         
        # Initialize the lattice as a 3D tensor with dimensions corresponding to
        # layers/orientations, width, and height. 
        self.lattice = torch.zeros((num_layers, width, height), dtype=torch.bool)

    def initialize_lattice(self, density):
        """
        Initialize the lattice with particles at a given density.

        :param density: Density of particles to be initialized.
        :type density: float
        """
        num_cells = self.width * self.height
        num_particles = int(density * num_cells)

        # Randomly place particles
        positions = np.random.choice(num_cells, num_particles, replace=False) # Randomly select positions by drawing num_particles samples from num_cells without replacement
        orientations = np.random.randint(0, ParticleLattice.NUM_ORIENTATIONS, num_particles)

        for pos, ori in zip(positions, orientations):
            x, y = divmod(pos, self.width) # Convert position to (x, y) coordinates. divmod returns quotient and remainder.
            self.lattice[ori, x, y] = True
 

    def add_particle(self, x, y, orientation):
        """
        Add a particle with a specific orientation at (x, y).

        :param x: x-coordinate of the lattice.
        :type x: int
        :param y: y-coordinate of the lattice.
        :type y: int
        :param orientation: Orientation of the particle.
        :type orientation: int
        """
        self.lattice[orientation, x, y] = True

    def remove_particle(self, x, y):
        """
        Remove a particle from a specific node in the lattice.

        :param x: x-coordinate of the node.
        :type x: int
        :param y: y-coordinate of the node.
        :type y: int
        """
        self.lattice[:, x, y] = 0  # Remove particle from all orientations

    def add_particle_flux(self, number_of_particles, region):
        """
        Add a number of particles randomly within a specified region.

        :param number_of_particles: Number of particles to add.
        :type number_of_particles: int
        param region: The region where particles are to be added, defined as (x_min, x_max, y_min, y_max).
        :type region: tuple
        """
        x_min, x_max, y_min, y_max = region
        for _ in range(number_of_particles):
            # Ensure that we add the particle in an empty spot
            while True:
                x = np.random.randint(x_min, x_max)
                y = np.random.randint(y_min, y_max)
                orientation = np.random.randint(0, self.num_layers)
                if not self.lattice[orientation, x, y]:  # Check if the spot is empty
                    self.lattice[orientation, x, y] = True
                    break

    def query_lattice_state(self):
        """
        Query the current state of the lattice.

        :return: The state of the lattice.
        :rtype: torch.Tensor
        """
        return self.lattice

    def compute_TM(self, v0):
        """
        Compute the migration transition rate tensor TM with periodic boundary conditions.
        """
        # Calculate empty cells (where no particle is present)
        empty_cells = ~self.lattice.sum(dim=0).bool()

        # Initialize the TM matrix to all zeros
        TM = torch.zeros(self.width, self.height, dtype=torch.float32)

        # Check if the particle can move in the direction it's facing
        # Up (orientation 0): Can move if the cell above is empty
        TM += v0 * self.lattice[0] * torch.roll(empty_cells, shifts=-1, dims=0)
        # Down (orientation 1): Can move if the cell below is empty
        TM += v0 * self.lattice[1] * torch.roll(empty_cells, shifts=1, dims=0)
        # Left (orientation 2): Can move if the cell to the left is empty
        TM += v0 * self.lattice[2] * torch.roll(empty_cells, shifts=-1, dims=1)
        # Right (orientation 3): Can move if the cell to the right is empty
        TM += v0 * self.lattice[3] * torch.roll(empty_cells, shifts=1, dims=1)

        print(TM)

        # Ensure TM remains binary by clipping its values to [0, 1]
        TM = TM.clamp(0, 1)


        return TM



    def compute_TR_tensor(self, g):
        """
        Compute the transition rate tensor TR for reorientation.

        :param g: Alignment sensitivity parameter.
        :type g: float
        """
        pass  # TODO: Implementation

    def get_statistics(self):
        """
        Compute various statistics of the lattice state.

        :return: Statistics of the current lattice state.
        :rtype: dict
        """
        pass  # TODO: Implementation

    def print_lattice(self):
        """
        Print the lattice with arrows indicating the orientations of the particles.
        """
        orientation_symbols = ["↑", "↓", "←", "→"]
        for y in range(self.height):
            row_str = ""
            for x in range(self.width):
                if self.lattice[:, x, y].any():
                    # Find which orientation(s) is/are present
                    symbols = [orientation_symbols[i] for i in range(self.num_layers) if self.lattice[i, x, y]]
                    row_str += "".join(symbols)
                else:
                    row_str += "·"  # Use a dot for empty cells
                row_str += " "  # Add space between cells
            print(row_str)

