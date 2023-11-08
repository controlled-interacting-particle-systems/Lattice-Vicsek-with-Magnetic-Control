import torch
import torch.nn.functional as F
import numpy as np


class ParticleLattice:
    """
    Class for the particle lattice.
    """

    NUM_ORIENTATIONS = 4  # Class level constant, shared by all instances of the class. Number of possible orientations for a particle
    def __init__(self, width: int, height: int, obstacles: torch.Tensor = None, sink: torch.Tensor = None):
        """
        Initialize the particle lattice.

        :param width: Width of the lattice.
        :param height: Height of the lattice.
        :param obstacles: A binary matrix indicating the obstacle locations.
        :param sink: A binary matrix indicating the sink (absorption) locations.
        """
    
        self.width = width
        self.height = height
        self.num_layers =ParticleLattice.NUM_ORIENTATIONS
         
        # Initialize the lattice as a 3D tensor with dimensions corresponding to
        # layers/orientations, width, and height. 
        self.lattice = torch.zeros((self.num_layers, height, width), dtype=torch.bool)

    # If an obstacles layer is provided, add it as an additional layer.
        if obstacles is not None:
            self.add_layer(obstacles)

        # If a sink layer is provided, add it as an additional layer.
        if sink is not None:
            self.add_layer(sink)

    def add_layer(self, layer: torch.Tensor):
        """
        Add a new layer to the lattice.

        :param layer: A binary matrix indicating the special cells for the new layer.
        """
        if layer.shape != (self.height, self.width):
            raise ValueError("Layer shape must match the dimensions of the lattice.")
        
        # Increment the number of layers
        self.num_layers += 1
        
        # Add the new layer to the lattice
        self.lattice = torch.cat((self.lattice, layer.unsqueeze(0)), dim=0)

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
            y, x = divmod(pos, self.width) # Convert position to (x, y) coordinates. divmod returns quotient and remainder.
            self.lattice[ori, y, x] = True
 

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
        self.lattice[:, y, x] = 0  # Remove particle from all orientations

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
                if not self.lattice[orientation, y, x]:  # Check if the spot is empty
                    self.lattice[orientation, y, x] = True
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

        # Calculate potential moves in each direction
        # Up
        TM_up = self.lattice[0] * empty_cells.roll(shifts=1, dims=0)
        # Down
        TM_down = self.lattice[1] * empty_cells.roll(shifts=-1, dims=0)
        # Left
        TM_left = self.lattice[2] * empty_cells.roll(shifts=1, dims=1)
        # Right
        TM_right = self.lattice[3] * empty_cells.roll(shifts=-1, dims=1)

        # Combine all moves
        TM = TM_up + TM_down + TM_left + TM_right

        return TM * v0


    def compute_TR(self, g):
        """
        Compute the reorientation transition rate tensor TR more efficiently.

        :param g: Parameter controlling alignment sensitivity. Default is 1.0.
        :type g: float
        """
        # Common kernel for convolution
        kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Convolve each orientation layer of the lattice
        TR_tensor = torch.zeros((4, self.width, self.height), dtype=torch.float32)
        for orientation in range(4):
            input_tensor = self.lattice[orientation].unsqueeze(0).unsqueeze(0).float()
            TR_tensor[orientation] = F.conv2d(input_tensor, kernel, padding=1)[0, 0]

        # Adjusting the TR tensor based on orientation vectors
        TR_tensor[0], TR_tensor[1] = TR_tensor[0] - TR_tensor[1], TR_tensor[1] - TR_tensor[0]
        TR_tensor[2], TR_tensor[3] = TR_tensor[2] - TR_tensor[3], TR_tensor[3] - TR_tensor[2]

        # Apply g and exponentiate
        TR_tensor *= g
        TR_tensor = torch.exp(TR_tensor)

        return TR_tensor
    def move_particle(self, x: int, y: int, new_x: int, new_y: int) -> bool:
        """
        Move a particle from (x, y) to (new_x, new_y).

        :param x: Current x-coordinate of the particle.
        :param y: Current y-coordinate of the particle.
        :param new_x: New x-coordinate of the particle.
        :param new_y: New y-coordinate of the particle.
        :return: True if the particle was moved successfully, False otherwise.
        """
        # Get the orientation of the particle at (x, y)
        orientation = self.lattice[:, y, x].nonzero(as_tuple=True)[0]

        # If no particle is found at the given location, return False
        if len(orientation) == 0:
            return False

        # If the target location is occupied or out of bounds, return False
        if (new_x < 0 or new_x >= self.width or
            new_y < 0 or new_y >= self.height or
            self.lattice[:, new_y, new_x].any()):
            return False

        # Move the particle
        self.lattice[orientation, new_y, new_x] = True
        self.lattice[orientation, y, x] = False

        return True
    
    def reorient_particle(self, x: int, y: int, new_orientation: int) -> bool:
        """
        Reorient a particle at (x, y) to a new orientation.

        :param x: x-coordinate of the particle.
        :param y: y-coordinate of the particle.
        :param new_orientation: The new orientation index for the particle.
        :return: True if the particle was reoriented successfully, False otherwise.
        """
        # Get the current orientation of the particle at (x, y)
        current_orientation = self.lattice[:, y, x].nonzero(as_tuple=True)[0]

        # If no particle is found at the given location, return False
        if len(current_orientation) == 0:
            return False

        # If the new orientation is the same as the current one, return False
        if current_orientation == new_orientation:
            return False
        
        # If the new orientation is out of bounds, return False
        if new_orientation < 0 or new_orientation >= ParticleLattice.NUM_ORIENTATIONS:
            return False

        # Reorient the particle
        self.lattice[current_orientation, y, x] = False
        self.lattice[new_orientation, y, x] = True

        return True

    def get_statistics(self) -> dict:
        """
        Compute various statistics of the lattice state.

        :return: Statistics of the current lattice state.
        :rtype: dict
        """
        # Sum only the first NUM_ORIENTATIONS layers to get the number of particles
        num_particles = self.lattice[:self.NUM_ORIENTATIONS].sum().item()
        density = num_particles / (self.width * self.height)  # Density of particles
        order_parameter = self.compute_order_parameter()  # Order parameter as defined before

        # Count the number of particles for each orientation
        orientation_counts = torch.sum(self.lattice[:self.NUM_ORIENTATIONS], dim=(1, 2)).tolist()

        stats = {
            'number_of_particles': num_particles,
            'density': density,
            'order_parameter': order_parameter,
            'orientation_counts': orientation_counts,
            # Include other statistics as needed
        }

        return stats

    def compute_order_parameter(self):
        """
        Compute the order parameter as the magnitude of the average orientation vector.

        :return: The order parameter of the lattice.
        :rtype: float
        """
        # Define unit vectors for each orientation
        orientation_vectors = torch.tensor([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=torch.float32)
        
        # Initialize the sum of orientation vectors
        orientation_vector_sum = torch.zeros(2, dtype=torch.float32)
        
        # Sum up orientation vectors for all particles
        for i, ori_vec in enumerate(orientation_vectors):
            num_particles = self.lattice[i].sum().item()  # Count number of particles with this orientation
            orientation_vector_sum += num_particles * ori_vec
        
        # Calculate the average orientation vector
        total_particles = self.lattice.sum().item()
        if total_particles == 0:
            return 0.0  # Avoid division by zero if there are no particles
        
        average_orientation_vector = orientation_vector_sum / total_particles
        
        # Calculate the magnitude of the average orientation vector
        order_parameter = torch.norm(average_orientation_vector, p=2)
        
        return order_parameter.item()

    def print_lattice(self):
        """
        Print the lattice with arrows indicating the orientations of the particles.
        """
        orientation_symbols = ["↑", "↓", "←", "→"]
        for y in range(self.height):
            row_str = ""
            for x in range(self.width):
                if self.lattice[:, y, x].any():
                    # Find which orientation(s) is/are present
                    symbols = [orientation_symbols[i] for i in range(self.num_layers) if self.lattice[i, y, x]]
                    row_str += "".join(symbols)
                else:
                    row_str += "·"  # Use a dot for empty cells
                row_str += " "  # Add space between cells
            print(row_str)

