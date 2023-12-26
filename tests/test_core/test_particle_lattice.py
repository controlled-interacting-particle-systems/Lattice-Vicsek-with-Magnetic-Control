import pytest
from lvmc.core.particle_lattice import ParticleLattice
import torch
import numpy as np


def test_particle_lattice_initialization():
    width, height, density = 10, 10, 0.5
    obstacles = torch.rand(height, width) > 0.5
    sinks = torch.rand(height, width) > 0.5

    lattice = ParticleLattice(width, height, density, obstacles, sinks)

    assert lattice.width == width
    assert lattice.height == height
    assert lattice.num_layers == lattice.NUM_ORIENTATIONS + (
        lattice.obstacles is not None
    ) + (lattice.sinks is not None)
    assert lattice.lattice.size() == (lattice.num_layers, height, width)
    assert all(name in lattice.layer_indices for name in lattice.ORIENTATION_LAYERS)
    assert "obstacles" in lattice.layer_indices
    assert "sinks" in lattice.layer_indices

    # Test density-based initialization (tbd later)
    # populated_cells = torch.sum(lattice.particles).item()
    # expected_cells = int(density * width * height * lattice.NUM_ORIENTATIONS)
    # assert populated_cells == expected_cells


def test_set_obstacle_success():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_obstacle(x, y)
    assert lattice.obstacles[y, x] == True


def test_set_sink_success():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 3, 3
    lattice.set_sink(x, y)
    assert lattice.is_sink(x, y) == True


def test_set_obstacle_on_occupied_cell():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 4, 4
    lattice.add_particle(x, y, 0)  # Add a particle

    # Attempt to set an obstacle on the same spot
    with pytest.raises(ValueError):
        lattice.set_obstacle(x, y)


def test_set_sink_on_obstacle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_obstacle(x, y)  # Set an obstacle

    # Attempt to set a sink on the same spot
    with pytest.raises(ValueError):
        lattice.set_sink(x, y)


def test_set_sink_on_occupied_cell():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 6, 6
    lattice.add_particle(x, y, 0)  # Set a sink

    # Attempt to set sink on the same spot
    with pytest.raises(ValueError):
        lattice.set_sink(x, y)


def test_set_obstacle_outside_bounds():
    lattice = ParticleLattice(width=10, height=10)
    # Coordinates outside the lattice bounds
    x, y = 11, 11
    with pytest.raises(ValueError):
        lattice.set_obstacle(x, y)


def test_set_sink_outside_bounds():
    lattice = ParticleLattice(width=10, height=10)
    # Coordinates outside the lattice bounds
    x, y = 11, 11
    with pytest.raises(ValueError):
        lattice.set_sink(x, y)


def test_add_layer():
    lattice = ParticleLattice(width=10, height=10)
    layer = torch.zeros((10, 10), dtype=torch.bool)
    lattice.add_layer(layer, "test_layer")
    assert "test_layer" in lattice.layer_indices
    assert lattice.num_layers == 7
    with pytest.warns(UserWarning):
        lattice.add_layer(layer, "test_layer")
        assert lattice.num_layers == 7
    # check if it raises and error when adding a layer with wrong dimensions
    with pytest.raises(ValueError):
        layer = torch.zeros((10, 11), dtype=torch.bool)
        lattice.add_layer(layer, "test_layer2")


def test_initialize_lattice():
    lattice = ParticleLattice(width=10, height=10)
    density = 0.5
    lattice.initialize_lattice(density)
    populated_cells = torch.sum(lattice.lattice).item()
    expected_cells = int(density * lattice.width * lattice.height)
    assert populated_cells == expected_cells
    # add test for obstacles and sinks


def test_is_empty():
    lattice = ParticleLattice(width=10, height=10)
    assert lattice.is_empty(5, 5) == True
    lattice.add_particle(5, 5, 0)
    assert lattice.is_empty(5, 5) == False


def test_add_particle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    orientation = 0
    lattice.add_particle(x, y, orientation)
    assert lattice.lattice[orientation, y, x] == True


def test_add_particle_outside_bounds():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 11, 11
    orientation = 0
    with pytest.raises(IndexError):
        lattice.add_particle(x, y, orientation)


def test_add_particle_on_obstacle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_obstacle(x, y)
    orientation = 0
    with pytest.raises(ValueError):
        lattice.add_particle(x, y, orientation)


def test_add_particle_on_sink():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_sink(x, y)
    orientation = 0
    # the particle should still be added
    lattice.add_particle(x, y, orientation)
    assert lattice.lattice[orientation, y, x] == True
    assert lattice.is_sink(x, y) == True


def test_add_particle_on_occupied_cell():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.add_particle(x, y, 0)
    orientation = 0
    with pytest.raises(ValueError):
        lattice.add_particle(x, y, orientation)


def test_remove_particle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.add_particle(x, y, 0)
    lattice.remove_particle(x, y)
    assert lattice.lattice[0, y, x] == False


def test_remove_particle_outside_bounds():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 11, 11
    with pytest.raises(IndexError):
        lattice.remove_particle(x, y)


def test_query_lattice_state():
    lattice = ParticleLattice(width=10, height=10)
    lattice.initialize_lattice(density=0.5)
    lattice_state = lattice.query_lattice_state()
    assert lattice_state.size() == (lattice.num_layers, lattice.height, lattice.width)
    assert torch.sum(lattice_state) == torch.sum(lattice.lattice)

def test_compute_tm():
    lattice = ParticleLattice(width=10, height=10)
    v0 = 1.0

    # add particles to the lattice
    lattice.add_particle(0, 0, 0)
    lattice.add_particle(0, 1, 1)
    lattice.add_particle(0, 2, 2)
    lattice.add_particle(0, 3, 3)

    # list of positions of the added particles
    positions = [(0, 0), (0, 1), (0, 2), (0, 3)]

    # compute the transition matrix
    tm = lattice.compute_tm(v0)

    # check if the transition matrix is of the correct shape
    assert tm.shape == (lattice.height, lattice.width)

    # check that the positions [(0,0), (0,1), (0,3)] are non zero entries in the transition matrix
    moving_positions = [(0, 0), (0, 1), (0, 3)]
    for position in moving_positions:
        assert tm[position[1], position[0]] == v0

    # check that the positions [(0,2)] are zero entries in the transition matrix
    stationary_positions = [(0, 2)]
    for position in stationary_positions:
        assert tm[position[1], position[0]] == 0

    # check that all the other entries in the transition matrix are zero
    zero_positions = [
        (0, 4),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 1),
        (2, 3),
        (2, 4),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 4),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    for position in zero_positions:
        assert tm[position[1], position[0]] == 0


def test_compute_tr():
    # Create a ParticleLattice instance
    lattice = ParticleLattice(width=10, height=10)
    g = 1.0
    tr = lattice.compute_tr(g)
    assert tr.shape == (lattice.NUM_ORIENTATIONS, lattice.height, lattice.width)

    # Check that an entry is non zero if and only if the the cell is occupied
    for x in range(lattice.width):
        for y in range(lattice.height):
            if lattice.is_empty(x, y):
                assert tr[:, y, x].sum() == 0
            else:
                for orientation in range(lattice.NUM_ORIENTATIONS):
                    assert tr[orientation, y, x] != 0


def test_compute_tr_with_obstacles():
    # Create a ParticleLattice instance
    lattice = ParticleLattice(width=10, height=10)
    g = 1.0
    lattice.set_obstacle(5, 5)
    tr = lattice.compute_tr(g)
    assert tr.shape == (lattice.NUM_ORIENTATIONS, lattice.height, lattice.width)

    # Check that an entry is non zero if and only if the the cell is occupied
    for x in range(lattice.width):
        for y in range(lattice.height):
            if lattice.is_empty(x, y):
                assert tr[:, y, x].sum() == 0
            else:
                for orientation in range(lattice.NUM_ORIENTATIONS):
                    assert tr[orientation, y, x] != 0


def test_compute_tr_with_sinks():
    # Create a ParticleLattice instance
    lattice = ParticleLattice(width=10, height=10)
    g = 1.0
    lattice.set_sink(5, 5)
    tr = lattice.compute_tr(g)
    assert tr.shape == (lattice.NUM_ORIENTATIONS, lattice.height, lattice.width)

    # Check that an entry is non zero if and only if the the cell is occupied
    for x in range(lattice.width):
        for y in range(lattice.height):
            if lattice.is_empty(x, y):
                assert tr[:, y, x].sum() == 0
            else:
                for orientation in range(lattice.NUM_ORIENTATIONS):
                    assert tr[orientation, y, x] != 0


def test_compute_tr_with_obstacles_and_sinks():
    # Create a ParticleLattice instance
    lattice = ParticleLattice(width=10, height=10)
    g = 1.0
    lattice.set_sink(5, 5)
    lattice.set_obstacle(5, 4)
    tr = lattice.compute_tr(g)
    assert tr.shape == (lattice.NUM_ORIENTATIONS, lattice.height, lattice.width)

    # Check that an entry is non zero if and only if the the cell is occupied
    for x in range(lattice.width):
        for y in range(lattice.height):
            if lattice.is_empty(x, y):
                assert tr[:, y, x].sum() == 0
            else:
                for orientation in range(lattice.NUM_ORIENTATIONS):
                    assert tr[orientation, y, x] != 0


def test_get_target_position():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5

    # Check if the target position is correct for each orientation
    # 1. Up
    assert lattice._get_target_position(x, y, lattice.layer_indices["up"]) == (x, y - 1)
    # 2. Down
    assert lattice._get_target_position(x, y, lattice.layer_indices["down"]) == (
        x,
        y + 1,
    )
    # 3. Left
    assert lattice._get_target_position(x, y, lattice.layer_indices["left"]) == (
        x - 1,
        y,
    )
    # 4. Right
    assert lattice._get_target_position(x, y, lattice.layer_indices["right"]) == (
        x + 1,
        y,
    )


def test_is_obstacle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_obstacle(x, y)
    assert lattice.is_obstacle(x, y)

    # Check if the method returns False for non-obstacle cells
    assert not lattice.is_obstacle(0, 0)
    assert not lattice.is_obstacle(1, 1)
    assert not lattice.is_obstacle(2, 2)


def test_is_sink():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_sink(x, y)
    assert lattice.is_sink(x, y)

    # Check if the method returns False for non-sink cells
    assert not lattice.is_sink(0, 0)
    assert not lattice.is_sink(1, 1)
    assert not lattice.is_sink(2, 2)


def test_is_sink_with_obstacles():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_sink(x, y)
    lattice.set_obstacle(x + 1, y)
    assert lattice.is_sink(x, y)

    # Check if the method returns False for non-sink cells
    assert not lattice.is_sink(0, 0)
    assert not lattice.is_sink(1, 1)
    assert not lattice.is_sink(2, 2)


def test_is_sink_with_particles():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_sink(x, y)
    lattice.add_particle(x, y, 0)

    assert lattice.is_sink(x, y)

    # Check if the method returns False for non-sink cells
    assert not lattice.is_sink(0, 0)
    assert not lattice.is_sink(1, 1)
    assert not lattice.is_sink(2, 2)


def test_get_particle_orientation():
    lattice = ParticleLattice(width=10, height=10)
    x, y = np.random.randint(0, lattice.width), np.random.randint(0, lattice.height)
    orientation = np.random.randint(0, 4)
    lattice.add_particle(x, y, orientation)
    assert lattice.get_particle_orientation(x, y) == orientation


def test_get_particle_orientation_outside_bounds():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 11, 11
    with pytest.raises(IndexError):
        lattice.get_particle_orientation(x, y)


def test_get_particle_orientation_on_empty_cell():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    with pytest.raises(ValueError):
        lattice.get_particle_orientation(x, y)


def test_get_particle_orientation_on_obstacle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_obstacle(x, y)
    with pytest.raises(ValueError):
        lattice.get_particle_orientation(x, y)


def test_get_particle_orientation_on_sink():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_sink(x, y)

    # If no particle is on the sink, it should raise an error
    with pytest.raises(ValueError):
        lattice.get_particle_orientation(x, y)

    # If a particle is on the sink, it should return the orientation of the particle
    orientation = np.random.randint(0, 4)
    lattice.add_particle(x, y, orientation)
    assert lattice.get_particle_orientation(x, y) == orientation


def test_get_particle_orientation_on_sink_with_obstacles():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_sink(x, y)
    lattice.set_obstacle(x + 1, y)

    # If no particle is on the sink, it should raise an error
    with pytest.raises(ValueError):
        lattice.get_particle_orientation(x, y)

    # If a particle is on the sink, it should return the orientation of the particle
    orientation = np.random.randint(0, 4)
    lattice.add_particle(x, y, orientation)
    assert lattice.get_particle_orientation(x, y) == orientation


def test_move_particle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    orientation = 0
    lattice.add_particle(x, y, orientation)
    lattice.move_particle(x, y)
    assert lattice.is_empty(x, y)
    assert not lattice.is_empty(x, y - 1)
    assert lattice.get_particle_orientation(x, y - 1) == orientation


def test_move_particle_outside_bounds():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 11, 11
    with pytest.raises(IndexError):
        lattice.move_particle(x, y)


def test_move_particle_on_empty_cell():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    with pytest.raises(ValueError):
        lattice.move_particle(x, y)


def test_move_particle_on_obstacle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_obstacle(x, y)
    with pytest.raises(ValueError):
        lattice.move_particle(x, y)


def test_move_particle_on_sink():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_sink(x, y)

    # If no particle is on the sink, it should raise an error
    with pytest.raises(ValueError):
        lattice.move_particle(x, y)

    # If a particle is on the sink, it should return the orientation of the particle
    orientation = np.random.randint(0, 4)
    lattice.add_particle(x, y, orientation)
    x_new, y_new = lattice._get_target_position(x, y, orientation)

    lattice.move_particle(x, y)
    assert lattice.is_empty(x, y)
    assert not lattice.is_empty(x_new, y_new)
    assert lattice.get_particle_orientation(x_new, y_new) == orientation


def test_move_particle_periodicity_up():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 0
    orientation = 0
    lattice.add_particle(x, y, orientation)
    lattice.move_particle(x, y)
    assert lattice.is_empty(x, y)
    assert not lattice.is_empty(x, y - 1)  # what is y-1? it's -1 so it's the last row
    assert lattice.get_particle_orientation(x, y - 1) == orientation


def test_move_particle_periodicity_down():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, lattice.height - 1
    orientation = lattice.layer_indices["down"]
    lattice.add_particle(x, y, orientation)
    lattice.move_particle(x, y)
    assert lattice.is_empty(x, y)
    assert not lattice.is_empty(x, 0)
    assert lattice.get_particle_orientation(x, 0) == orientation


def test_move_particle_periodicity_left():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 0, 5
    orientation = lattice.layer_indices["left"]
    lattice.add_particle(x, y, orientation)
    lattice.move_particle(x, y)
    assert lattice.is_empty(x, y)
    assert not lattice.is_empty(lattice.width - 1, y)
    assert lattice.get_particle_orientation(lattice.width - 1, y) == orientation


def test_move_particle_periodicity_right():
    lattice = ParticleLattice(width=10, height=10)
    x, y = lattice.width - 1, 5
    orientation = lattice.layer_indices["right"]
    lattice.add_particle(x, y, orientation)
    lattice.move_particle(x, y)
    assert lattice.is_empty(x, y)
    assert not lattice.is_empty(0, y)
    assert lattice.get_particle_orientation(0, y) == orientation


def test_reorient_particle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    orientation = np.random.randint(0, 4)
    lattice.add_particle(x, y, orientation)
    new_orientation = np.random.randint(0, 4)
    lattice.reorient_particle(x, y, new_orientation)
    assert lattice.get_particle_orientation(x, y) == new_orientation
    # check that the particle is still there
    assert not lattice.is_empty(x, y)
    # attempt to reorient to the same orientation
    lattice.reorient_particle(x, y, new_orientation)
    assert lattice.get_particle_orientation(x, y) == new_orientation
    # attempt to reorient to an invalid orientation
    with pytest.raises(IndexError):
        lattice.reorient_particle(x, y, 4)
    # attempt to reorient a non-existent particle
    with pytest.raises(ValueError):
        lattice.reorient_particle(0, 0, 0)


