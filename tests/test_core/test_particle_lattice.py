import pytest
from particle_lattice.core.particle_lattice import ParticleLattice
import torch

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
    region_particles = lattice.query_lattice_state()[:, x_min:x_max, y_min:y_max].sum().item()

    assert region_particles == number_of_particles
    # Additionally, you can add checks to ensure particles are only within the specified region

def test_compute_TM_with_periodic_boundaries():
    # Initialize a 5x5 lattice for testing
    width, height, num_layers = 5, 5, 4
    lattice = ParticleLattice(width=width, height=height, num_layers=num_layers)
    lattice.initialize_lattice(density=0)
    v0 = 1.0

    # Test cases: (position, orientation, expected position after move)
    test_cases = [
        ((0, 0), 0, (4, 0)),   # Top edge, moving up
        ((4, 0), 1, (0, 0)),   # Bottom edge, moving down
        ((0, 0), 2, (0, 4)),   # Left edge, moving left
        ((0, 4), 3, (0, 0)),   # Right edge, moving right
        ((0, 0), 3, (0, 1)),   # Non-edge case, moving right
    ]

    for (x, y), orientation, (exp_x, exp_y) in test_cases:
        # Reset lattice and add a single particle
        lattice.initialize_lattice(density=0)
        lattice.add_particle(x, y, orientation)

        # Compute TM
        TM = lattice.compute_TM(v0)
        lattice.print_lattice()
        print(TM)

        # Check if the particle can move to the expected position
        assert TM[x, y] == v0, f"Particle at ({x}, {y}) with orientation {orientation} should be able to move to ({exp_x}, {exp_y})"
