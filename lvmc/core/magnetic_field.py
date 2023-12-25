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

    def apply(self, lattice):
        """
        Apply the magnetic field to all particles on the lattice (a 90 degrees rotation in the prescribed direction).

        :param lattice: The lattice object.
        :type lattice: Lattice
        """
        # the lattice is a 3D binary tensor with dimensions corresponding to layers/orientations, width, and height. orientation layers are up, down, left, and right in that order.

        # rotate the lattice by 90 degrees in the prescribed direction

        # rotating the lattice is equivalent to shuffling the orientation layers

        lattice.lattice[...] = lattice.lattice.roll(self.current_direction, dims=0)
