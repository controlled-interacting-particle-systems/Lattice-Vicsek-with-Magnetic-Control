import numpy as np
from particle_lattice.core.particle import Particle

class Lattice:
    """
    This class manages the 2D lattice with periodic boundary conditions.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.empty((self.height, self.width), dtype=object)
        self.particles = [] # list of particles

    def initialize_lattice(self):
        self.grid.fill(None)

    def get_nearest_neighbors(self, x, y):
        neighbors = [
            ((x + 1) % self.width, y),
            ((x - 1) % self.width, y),
            (x, (y + 1) % self.height),
            (x, (y - 1) % self.height)
        ]
        return neighbors

    def is_node_empty(self, x, y):
        return self.grid[y, x] is None
    
    def place_particle(self, particle: Particle, x: int, y: int):
        """
        Place a particle on a specific node in the lattice.

        :param particle: Particle object to be placed.
        :type particle: Particle
        :param x: x-coordinate of the node.
        :type x: int
        :param y: y-coordinate of the node.
        :type y: int
        """
        self.grid[y, x] = particle

