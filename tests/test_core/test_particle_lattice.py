import pytest
from lvmc.core.particle_lattice import ParticleLattice, Orientation
import torch
import numpy as np


def test_particle_lattice_initialization():
    width, height, density = 10, 10, 0.5
    lattice = ParticleLattice(width, height, density)

    assert lattice.width == width
    assert lattice.height == height


def test_set_obstacle_success():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_obstacle(x, y)
    assert lattice.obstacles[y, x] == True


def test_set_sink_success():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 3, 3
    lattice.set_sink(x, y)
    assert lattice._is_sink(x, y) == True


def test_set_obstacle_on_occupied_cell():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 4, 4
    lattice.add_particle(x, y, Orientation.UP)  # Add a particle

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
    lattice.add_particle(x, y, Orientation.UP)  # Set a sink

    # Attempt to set sink on the same spot
    with pytest.raises(ValueError):
        lattice.set_sink(x, y)


def test_set_obstacle_outside_bounds():
    lattice = ParticleLattice(width=10, height=10)
    # Coordinates outside the lattice bounds
    x, y = 11, 11
    with pytest.raises(IndexError):
        lattice.set_obstacle(x, y)


def test_set_sink_outside_bounds():
    lattice = ParticleLattice(width=10, height=10)
    # Coordinates outside the lattice bounds
    x, y = 11, 11
    with pytest.raises(IndexError):
        lattice.set_sink(x, y)


def test__initialize_lattice():
    lattice = ParticleLattice(width=10, height=10)
    density = 0.5
    lattice._initialize_lattice(density)
    populated_cells = torch.sum(lattice.particles).item()
    expected_cells = int(density * lattice.width * lattice.height)
    assert populated_cells == expected_cells
    # add test for obstacles and sinks


def test__is_empty():
    lattice = ParticleLattice(width=10, height=10)
    assert lattice._is_empty(5, 5) == True
    lattice.add_particle(5, 5, Orientation.UP)
    assert lattice._is_empty(5, 5) == False


def test_add_particle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    orientation = Orientation.UP  # Changed to use Orientation enum
    lattice.add_particle(x, y, orientation)
    assert lattice.particles[orientation.value, y, x] == True


def test_add_particle_outside_bounds():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 11, 11
    orientation = Orientation.UP
    with pytest.raises(IndexError):
        lattice.add_particle(x, y, orientation)


def test_add_particle_on_obstacle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_obstacle(x, y)
    orientation = Orientation.UP
    with pytest.raises(ValueError):
        lattice.add_particle(x, y, orientation)


def test_add_particle_on_sink():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_sink(x, y)
    orientation = Orientation.UP
    # the particle should still be added
    lattice.add_particle(x, y, orientation)
    assert lattice.particles[orientation.value, y, x] == True
    assert lattice._is_sink(x, y) == True


def test_add_particle_on_occupied_cell():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    orientation = Orientation.UP
    lattice.add_particle(x, y, orientation)

    with pytest.raises(ValueError):
        lattice.add_particle(x, y, orientation)


def test_remove_particle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    orientation = Orientation.UP
    lattice.add_particle(x, y, orientation)
    lattice.remove_particle(x, y)
    assert lattice.particles[orientation.value, y, x] == False


def test_remove_particle_outside_bounds():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 11, 11
    with pytest.raises(IndexError):
        lattice.remove_particle(x, y)


def test_query_lattice_state():
    lattice = ParticleLattice(width=10, height=10)
    lattice._initialize_lattice(density=0.5)
    lattice_state = lattice.query_lattice_state()
    assert lattice_state.size() == (
        lattice.NUM_ORIENTATIONS,
        lattice.height,
        lattice.width,
    )
    assert torch.sum(lattice_state) == torch.sum(lattice.particles)


def test_compute_tm():
    lattice = ParticleLattice(width=10, height=10)
    v0 = 1.0

    positions = [(0, 0), (0, 1), (0, 2), (0, 3)]

    # add particles to the lattice
    for position, ori in zip(positions, Orientation):
        lattice.add_particle(position[0], position[1], ori)

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
            if lattice._is_empty(x, y):
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
            if lattice._is_empty(x, y):
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
            if lattice._is_empty(x, y):
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
            if lattice._is_empty(x, y):
                assert tr[:, y, x].sum() == 0
            else:
                for orientation in range(lattice.NUM_ORIENTATIONS):
                    assert tr[orientation, y, x] != 0


def test_get_target_position():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5

    # Add a particle to the lattice in the cell (x,y)
    orientation = np.random.choice(list(Orientation))
    lattice.add_particle(x, y, orientation)

    # Check if the target position is correct for each orientation
    # 1. Up
    assert lattice._get_target_position(x, y, Orientation.UP) == (
        x,
        (y - 1) % lattice.height,
    )
    # 2. Down
    assert lattice._get_target_position(x, y, Orientation.DOWN) == (
        x,
        (y + 1) % lattice.height,
    )
    # 3. Left
    assert lattice._get_target_position(x, y, Orientation.LEFT) == (
        (x - 1) % lattice.width,
        y,
    )
    # 4. Right
    assert lattice._get_target_position(x, y, Orientation.RIGHT) == (
        (x + 1) % lattice.width,
        y,
    )


def test__is_obstacle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_obstacle(x, y)
    assert lattice._is_obstacle(x, y)

    # Check if the method returns False for non-obstacle cells
    assert not lattice._is_obstacle(0, 0)
    assert not lattice._is_obstacle(1, 1)
    assert not lattice._is_obstacle(2, 2)


def test__is_sink():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_sink(x, y)
    assert lattice._is_sink(x, y)

    # Check if the method returns False for non-sink cells
    assert not lattice._is_sink(0, 0)
    assert not lattice._is_sink(1, 1)
    assert not lattice._is_sink(2, 2)


def test__is_sink_with_obstacles():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    lattice.set_sink(x, y)
    lattice.set_obstacle(x + 1, y)
    assert lattice._is_sink(x, y)

    # Check if the method returns False for non-sink cells
    assert not lattice._is_sink(0, 0)
    assert not lattice._is_sink(1, 1)
    assert not lattice._is_sink(2, 2)


def test__is_sink_with_particles():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    orientation = Orientation.UP
    lattice.set_sink(x, y)
    lattice.add_particle(x, y, orientation)

    assert lattice._is_sink(x, y)

    # Check if the method returns False for non-sink cells
    assert not lattice._is_sink(0, 0)
    assert not lattice._is_sink(1, 1)
    assert not lattice._is_sink(2, 2)


def test_get_particle_orientation():
    lattice = ParticleLattice(width=10, height=10)
    x, y = np.random.randint(0, lattice.width), np.random.randint(0, lattice.height)
    orientation = Orientation(np.random.randint(0, len(Orientation)))
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
    orientation = np.random.choice(list(Orientation))
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
    orientation = np.random.choice(list(Orientation))
    lattice.add_particle(x, y, orientation)
    assert lattice.get_particle_orientation(x, y) == orientation


def test_move_particle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    orientation = Orientation.UP
    lattice.add_particle(x, y, orientation)
    successful_move = lattice.move_particle(x, y)

    assert successful_move
    assert lattice._is_empty(x, y)
    assert not lattice._is_empty(x, (y - 1) % lattice.height)
    assert lattice.get_particle_orientation(x, (y - 1) % lattice.height) == orientation

    # Attempt to move a particle in an empty cell
    with pytest.raises(ValueError):
        lattice.move_particle(0, 0)


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
    orientation = np.random.choice(list(Orientation))
    lattice.add_particle(x, y, orientation)
    x_new, y_new = lattice._get_target_position(x, y, orientation)

    lattice.move_particle(x, y)
    assert lattice._is_empty(x, y)
    assert not lattice._is_empty(x_new, y_new)
    assert lattice.get_particle_orientation(x_new, y_new) == orientation


def test_move_particle_periodicity_up():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 0
    orientation = Orientation.UP
    lattice.add_particle(x, y, orientation)
    lattice.move_particle(x, y)

    assert lattice._is_empty(x, y)
    assert not lattice._is_empty(x, lattice.height - 1)
    assert lattice.get_particle_orientation(x, lattice.height - 1) == orientation


def test_move_particle_periodicity_down():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, lattice.height - 1
    orientation = Orientation.DOWN
    lattice.add_particle(x, y, orientation)
    lattice.move_particle(x, y)

    assert lattice._is_empty(x, y)
    assert not lattice._is_empty(x, 0)
    assert lattice.get_particle_orientation(x, 0) == orientation


def test_move_particle_periodicity_left():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 0, 5
    orientation = Orientation.LEFT
    lattice.add_particle(x, y, orientation)
    lattice.move_particle(x, y)

    assert lattice._is_empty(x, y)
    assert not lattice._is_empty(lattice.width - 1, y)
    assert lattice.get_particle_orientation(lattice.width - 1, y) == orientation


def test_move_particle_periodicity_right():
    lattice = ParticleLattice(width=10, height=10)
    x, y = lattice.width - 1, 5
    orientation = Orientation.RIGHT
    lattice.add_particle(x, y, orientation)
    lattice.move_particle(x, y)

    assert lattice._is_empty(x, y)
    assert not lattice._is_empty(0, y)
    assert lattice.get_particle_orientation(0, y) == orientation


def test_reorient_particle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    original_orientation = Orientation(np.random.choice(list(Orientation)))
    lattice.add_particle(x, y, original_orientation)

    # Choose a new orientation different from the original
    new_orientations = list(set(Orientation) - {original_orientation})
    new_orientation = np.random.choice(new_orientations)

    lattice.reorient_particle(x, y, new_orientation)
    assert lattice.get_particle_orientation(x, y) == new_orientation

    # Check reorientation to the same orientation
    assert not lattice.reorient_particle(x, y, new_orientation)

    # Attempt to reorient a non-existent particle
    with pytest.raises(ValueError):
        lattice.reorient_particle(0, 0, Orientation.UP)


def test_transport_particle():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    original_orientation = Orientation(np.random.choice(list(Orientation)))
    lattice.add_particle(x, y, original_orientation)

    # Choose a new orientation different from the original
    new_orientations = list(set(Orientation) - {original_orientation})
    direction = np.random.choice(new_orientations)

    # Get the target position
    x_new, y_new = lattice._get_target_position(x, y, direction)

    lattice.transport_particle(x, y, direction)
    assert lattice._is_empty(x, y)
    # Check that the new position is not empty

    assert not lattice._is_empty(x_new, y_new)

    # Check that the orientation of the particle is correct
    assert lattice.get_particle_orientation(x_new, y_new) == original_orientation


def test_compute_log_tr_obstacles():
    lattice = ParticleLattice(width=10, height=10)
    x, y = 5, 5
    orientation = np.random.choice(list(Orientation))
    lattice.add_particle(x, y, orientation)

    # Get the target position
    x_new, y_new = lattice._get_target_position(x, y, orientation)

    # Add an obstacle at the target position
    lattice.set_obstacle(x_new, y_new)

    # Compute the log transition rate term corresponding to the obstacle
    log_tr = lattice.compute_log_tr_obstacles()

    up_down_log_rates = 0.5 * torch.tensor([0.0, 1.0, 0.0, 1.0])
    left_right_log_rates = 0.5 * torch.tensor([1.0, 0.0, 1.0, 0.0])
    if orientation == Orientation.UP or orientation == Orientation.DOWN:
        assert torch.allclose(log_tr[:, y, x], up_down_log_rates)
    else:
        assert torch.allclose(log_tr[:, y, x], left_right_log_rates)

    # Check that all other x,y entries are zero
    for x in range(lattice.width):
        for y in range(lattice.height):
            if x != x_new and y != y_new:
                assert log_tr[:, y, x].sum() == 0.0
