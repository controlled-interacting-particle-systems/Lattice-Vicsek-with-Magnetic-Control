class MagneticField:
    """
    Class for managing magnetic field effects on particles.
    """

    def __init__(self, initial_direction=0):
        """
        Initialize MagneticField with an initial direction.

        :param initial_direction: One of -1, 0, or 1
            - -1: Counterclockwise
            - 0: None
            - 1: Clockwise
        :type initial_direction: int
        """
        self.current_direction = initial_direction

    def set_direction(self, direction):
        """
        Set the current direction of the magnetic field.

        :param direction: One of -1, 0, or 1
        :type direction: int
        """
        self.current_direction = direction

    def rotate_particle(self, particle):
        """
        Rotate the particle's orientation based on the magnetic field direction.

        :param particle: The particle to rotate.
        :type particle: Particle
        """
        particle.set_orientation(
            (particle.get_orientation() - self.current_direction) % 4
        )

    def apply(self, lattice):
        """
        Apply the magnetic field to all particles on the lattice.

        :param lattice: The lattice object.
        :type lattice: Lattice
        """
        for y in range(lattice.height):
            for x in range(lattice.width):
                particle = lattice.grid[y, x]
                if particle is not None:
                    self.rotate_particle(particle)
