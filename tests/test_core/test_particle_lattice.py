import pytest
from vlmc.core.particle_lattice import ParticleLattice
import torch



def test_particle_lattice_initialization():
    width, height, density = 10, 10, 0.5
    obstacles = torch.rand(height, width) > 0.5
    sinks = torch.rand(height, width) > 0.5

    lattice = ParticleLattice(width, height, density, obstacles, sinks)

    assert lattice.width == width
    assert lattice.height == height
    assert lattice.num_layers == lattice.NUM_ORIENTATIONS + (lattice.obstacles is not None) + (lattice.sinks is not None)
    assert lattice.lattice.size() == (lattice.num_layers, height, width)
    assert all(name in lattice.layer_indices for name in lattice.ORIENTATION_LAYERS)
    assert 'obstacles' in lattice.layer_indices
    assert 'sinks' in lattice.layer_indices

    # Test density-based initialization (tbd later)
    # populated_cells = torch.sum(lattice.particles).item()
    # expected_cells = int(density * width * height * lattice.NUM_ORIENTATIONS)
    # assert populated_cells == expected_cells

    




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
