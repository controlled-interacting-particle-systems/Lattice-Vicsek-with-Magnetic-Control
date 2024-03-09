import pytest
from lvmc.core.particle_lattice import ParticleLattice, Orientation
from lvmc.core.magnetic_field import MagneticField


class TestMagneticField:
    @pytest.fixture
    def lattice(self):
        # Mock a ParticleLattice object
        return ParticleLattice(width=5, height=3)

    def test_initialization(self):
        for direction in [-1, 0, 1]:
            field = MagneticField(initial_direction=direction)
            assert (
                field.get_current_direction() == direction
            ), "Initial direction not set correctly"

    def test_set_direction(self):
        field = MagneticField()
        for direction in [-1, 0, 1]:
            field.set_direction(direction)
            assert (
                field.get_current_direction() == direction
            ), "Setting direction failed"

    def test_apply(self, lattice):
        clockwise_rotation = {
            Orientation.UP: Orientation.RIGHT,
            Orientation.RIGHT: Orientation.DOWN,
            Orientation.DOWN: Orientation.LEFT,
            Orientation.LEFT: Orientation.UP,
        }

        counterclockwise_rotation = {v: k for k, v in clockwise_rotation.items()}

        field = MagneticField()
        # Apply clockwise rotation
        field.set_direction(-1)
        pos_to_orientation = {
            pos: lattice.get_particle_orientation(*pos)
            for pos in lattice.position_to_particle_id.keys()
        }
        field.apply(lattice)
        for pos, orientation in pos_to_orientation.items():
            assert (
                lattice.get_particle_orientation(*pos)
                == clockwise_rotation[orientation]
            ), "Clockwise rotation failed"

    def test_get_current_direction(self):
        field = MagneticField()
        for direction in [-1, 0, 1]:
            field.set_direction(direction)
            assert (
                field.get_current_direction() == direction
            ), "Getting current direction failed"
