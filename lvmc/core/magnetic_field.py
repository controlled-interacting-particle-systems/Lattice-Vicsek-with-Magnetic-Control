from lvmc.core.particle_lattice import ParticleLattice  # for type hinting


class MagneticField:
    """
    Class for managing magnetic field effects on particles.
    """

    def __init__(self, initial_direction: int = 0):
        """
        Initialize MagneticField with an initial direction.

        :param initial_direction: One of -1, 0, or 1
            - -1: Clockwise
            - 0: None
            - 1: Counterclockwise
        :type initial_direction: int
        """
        self.current_direction = initial_direction

    def set_direction(self, direction: int) -> None:
        """
        Set the current direction of the magnetic field.

        :param direction: One of -1, 0, or 1
        :type direction: int
        """
        self.current_direction = direction

    def apply(self, lattice: ParticleLattice) -> None:
        """
        Apply the magnetic field to all particles on the lattice (a 90 degrees rotation in the prescribed direction).
        :param lattice: The lattice object.
        :type lattice: ParticleLattice
        """
        # Rotate the lattice by 90 degrees in the prescribed direction, using numpy.roll
        lattice.particles[...] = lattice.particles.roll(self.current_direction, dims=0) 

        # Function to safely get the value from an Orientation or return a placeholder
        def get_orientation_value(orientation):
            return -1 if orientation is None else orientation.value

        # Vectorize the function
        get_value_vectorized = np.vectorize(get_orientation_value)

        # Apply the function to the orientation map
        numeric_orientation_map = get_value_vectorized(lattice.orientation_map)

        # Apply rotation
        if self.current_direction == 1:  # Clockwise rotation
            numeric_orientation_map = (numeric_orientation_map + 1) % 4
        elif self.current_direction == -1:  # Counterclockwise rotation
            numeric_orientation_map = (numeric_orientation_map - 1) % 4

        # Convert back to Orientation enum, handling the placeholder
        lattice.orientation_map = np.vectorize(
            lambda x: None if x == -1 else Orientation(x)
        )(numeric_orientation_map)

    def get_current_direction(self) -> int:
        """
        Get the current direction of the magnetic field.

        :return: int - The current direction of the magnetic field.
        """
        return self.current_direction
