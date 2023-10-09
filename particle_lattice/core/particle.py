import numpy as np
import math

# Dictionary to map integer codes to their NumPy array vector representations
ORIENTATION_TO_VECTOR = {
    0: np.array([0, 1]),  # Up
    1: np.array([0, -1]), # Down
    2: np.array([-1, 0]), # Left
    3: np.array([1, 0])   # Right
}

class Particle:
    """
    Class to manage individual particle states and transition rates.
    
    Attributes:
        x (int): x-coordinate of the particle on the lattice.
        y (int): y-coordinate of the particle on the lattice.
        orientation_code (int): Integer code representing the orientation.
    """

    def __init__(self, x: int, y: int, orientation_code: int):
        """
        Initialize a Particle instance.
        
        :param x: x-coordinate of the particle.
        :type x: int
        :param y: y-coordinate of the particle.
        :type y: int
        :param orientation_code: Integer code representing the orientation.
        :type orientation_code: int
        """
        self.x = x
        self.y = y
        self.orientation_code = orientation_code

    def set_orientation(self, orientation_code: int):
        """
        Set the orientation of the particle.
        
        :param orientation_code: New orientation code.
        :type orientation_code: int
        """
        self.orientation_code = orientation_code

    def get_orientation(self) -> int:
        """
        Get the orientation code of the particle.
        
        :return: Current orientation code.
        :rtype: int
        """
        return self.orientation_code

    def get_orientation_vector(self) -> np.ndarray:
        """
        Get the vector representation of the orientation.
        
        :return: Vector representing the orientation.
        :rtype: np.ndarray
        """
        return ORIENTATION_TO_VECTOR[self.orientation_code]

    def move(self, x: int, y: int):
        """
        Move the particle to a new position.
        
        :param x: New x-coordinate.
        :type x: int
        :param y: New y-coordinate.
        :type y: int
        """
        self.x = x
        self.y = y

    def reorient(self, new_orientation_code: int):
        """
        Reorient the particle to a new orientation.
        
        :param new_orientation_code: New orientation code.
        :type new_orientation_code: int
        """
        self.orientation_code = new_orientation_code

    def is_destination_empty(self, lattice: 'Lattice') -> bool:
        """
        Check if the destination node based on current position and orientation is empty.
        
        :param lattice: Lattice object.
        :type lattice: Lattice
        :return: True if destination is empty, False otherwise.
        :rtype: bool
        """
        dx, dy = self.get_orientation_vector()
        new_x, new_y = (self.x + dx) % lattice.width, (self.y + dy) % lattice.height
        return lattice.is_node_empty(new_x, new_y)

    def compute_TR(self, lattice, new_orientation_code: int, g: float) -> float:
        """
        Compute the reorientation transition rate (TR).
        
        :param lattice: Lattice object.
        :type lattice: Lattice
        :param new_orientation_code: New orientation code.
        :type new_orientation_code: int
        :param g: Parameter controlling alignment sensitivity.
        :type g: float
        :return: Computed TR value.
        :rtype: float
        """
        neighbors = lattice.get_nearest_neighbors(self.x, self.y)
        new_vector = ORIENTATION_TO_VECTOR[new_orientation_code]

        sum_term = 0
        for nx, ny in neighbors:
            neighbor_particle = lattice.grid[ny, nx]
            if neighbor_particle is not None:
                neighbor_vector = neighbor_particle.get_orientation_vector()
                sum_term += np.dot(new_vector, neighbor_vector)

        return math.exp(g * sum_term)

    def compute_TM(self, lattice, v0: float) -> float:
        """
        Compute the migration transition rate (TM).
        
        :param lattice: Lattice object.
        :type lattice: Lattice
        :param v0: Base rate for migration.
        :type v0: float
        :return: Computed TM value.
        :rtype: float
        """
        if self.is_destination_empty(lattice):
            return v0
        else:
            return 0
