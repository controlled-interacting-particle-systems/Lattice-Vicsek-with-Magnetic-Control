import pytest
from vlmc.core.particle_lattice import ParticleLattice
import torch


def test_initialize_lattice():
    width, height, num_layers = 10, 10, 4
    lattice = ParticleLattice(width, height)

    # Test for successful initialization at a given density
    density = 1
    lattice.initialize_lattice(density)
    actual_density = lattice.lattice.float().mean().item() * num_layers
    assert actual_density == pytest.approx(
        density, abs=0.1
    ), f"Expected density approximately {density}, but got {actual_density}"


def test_add_remove_particle():
    lattice = ParticleLattice(width=10, height=10)
    lattice.add_particle(5, 5, 0)

    # Check if particle is added
    assert lattice.query_lattice_state()[0, 5, 5] == 1

    lattice.remove_particle(5, 5)

    # Check if particle is removed
    assert lattice.query_lattice_state().sum().item() == 0


def test_add_particle_flux():
    lattice = ParticleLattice(width=20, height=20)
    number_of_particles = 5
    region = (5, 10, 5, 10)  # Define a region within the lattice

    lattice.add_particle_flux(number_of_particles, region)
    x_min, x_max, y_min, y_max = region

    # Count the number of particles in the region
    region_particles = (
        lattice.query_lattice_state()[:, x_min:x_max, y_min:y_max].sum().item()
    )

    assert region_particles == number_of_particles
    # Additionally, you can add checks to ensure particles are only within the specified region


def test_init_without_additional_layers():
    width, height = 10, 5
    lattice = ParticleLattice(width, height)

    assert lattice.width == width
    assert lattice.height == height
    assert lattice.num_layers == ParticleLattice.NUM_ORIENTATIONS
    assert lattice.lattice.shape == (ParticleLattice.NUM_ORIENTATIONS, height, width)
    assert torch.all(lattice.lattice == 0)


def test_init_with_obstacles_layer():
    width, height = 10, 5
    obstacles = torch.ones((height, width), dtype=torch.bool)
    lattice = ParticleLattice(width, height, obstacles=obstacles)

    assert lattice.num_layers == ParticleLattice.NUM_ORIENTATIONS + 1
    assert torch.all(lattice.lattice[-1, :, :] == obstacles)


def test_add_layer():
    width, height = 10, 5
    lattice = ParticleLattice(width, height)
    initial_num_layers = lattice.num_layers

    new_layer = torch.ones((height, width), dtype=torch.bool)
    lattice.add_layer(new_layer)

    assert lattice.num_layers == initial_num_layers + 1
    assert torch.all(lattice.lattice[-1, :, :] == new_layer)


def test_add_layer_incorrect_shape():
    width, height = 10, 5
    lattice = ParticleLattice(width, height)
    new_layer = torch.ones((height + 1, width), dtype=torch.bool)  # Incorrect shape

    try:
        lattice.add_layer(new_layer)
    except ValueError as e:
        assert str(e) == "Layer shape must match the dimensions of the lattice."
    else:
        assert False, "ValueError not raised"


def test_print_lattice(capsys):
    # Create a small lattice for testing
    width, height = 3, 4
    lattice = ParticleLattice(height, width)

    # Manually set the lattice to a known state
    lattice.lattice[0, 0, 0] = True  # Up arrow at (0,0)
    lattice.lattice[1, 0, 1] = True  # Down arrow at (0,1)
    lattice.lattice[2, 1, 0] = True  # Left arrow at (1,0)
    lattice.lattice[3, 1, 1] = True  # Right arrow at (1,1)

    # Call the print_lattice function
    lattice.print_lattice()

    # Capture the output
    captured = capsys.readouterr()

    # Define the expected output
    expected_output = "↑ ↓ · · \n← → · · \n· · · · \n"

    # Assert that the expected output matches the captured output
    assert captured.out == expected_output


def test_move_particle():
    lattice = ParticleLattice(width=5, height=5)
    lattice.add_particle(2, 2, 0)  # Add particle at the center with orientation 0
    assert lattice.move_particle(2, 2, 3, 3) == True  # Attempt to move to an empty spot
    assert lattice.lattice[0, 3, 3] == True  # Check if the particle has moved
    assert lattice.lattice[0, 2, 2] == False  # Check if the original spot is now empty
    assert (
        lattice.move_particle(3, 3, 3, 3) == False
    )  # Attempt to move to the same spot
    assert lattice.move_particle(3, 3, 5, 5) == False  # Attempt to move out of bounds


def test_reorient_particle():
    lattice = ParticleLattice(width=5, height=5)
    lattice.add_particle(2, 2, 0)  # Add particle at the center with orientation 0
    assert lattice.reorient_particle(2, 2, 1) == True  # Change orientation to 1
    assert lattice.lattice[1, 2, 2] == True  # Check if the particle has new orientation
    assert (
        lattice.lattice[0, 2, 2] == False
    )  # Check if the particle no longer has the old orientation
    assert (
        lattice.reorient_particle(2, 2, 1) == False
    )  # Attempt to reorient to the same orientation
    assert (
        lattice.reorient_particle(2, 2, 4) == False
    )  # Attempt to reorient to an invalid orientation (out of bounds)
    assert (
        lattice.reorient_particle(3, 3, 2) == False
    )  # Attempt to reorient a non-existent particle


def test_get_statistics():
    lattice = ParticleLattice(width=10, height=10)
    lattice.initialize_lattice(density=0.5)  # Initialize with a density of 0.5

    stats = lattice.get_statistics()

    assert "number_of_particles" in stats
    assert "density" in stats
    assert "order_parameter" in stats
    assert "orientation_counts" in stats
    assert stats["density"] == pytest.approx(0.5)
    assert stats["number_of_particles"] == pytest.approx(100 * 0.5)
    assert (
        0 <= stats["order_parameter"] <= 1
    )  # Order parameter should be between 0 and 1

    # Check if orientation counts is a list with four elements (one for each orientation)
    assert isinstance(stats["orientation_counts"], list)
    assert len(stats["orientation_counts"]) == ParticleLattice.NUM_ORIENTATIONS
    # Check if the sum of the orientation counts equals the total number of particles
    assert sum(stats["orientation_counts"]) == stats["number_of_particles"]


def test_compute_order_parameter():
    # Create a ParticleLattice instance
    lattice = ParticleLattice(width=10, height=10)

    # Manually set a known lattice configuration that would result in a non-zero order parameter
    # For simplicity, let's say all particles are oriented up
    lattice.lattice[0] = torch.ones((10, 10), dtype=torch.bool)

    # Call the compute_order_parameter method
    order_parameter = lattice.compute_order_parameter()

    # Since all particles are aligned, the order parameter should be 1
    assert order_parameter == pytest.approx(1.0)
