from lvmc.core.particle_lattice import Orientation, ParticleLattice
from lvmc.core.flow import Flow, PoiseuilleFlow
from icecream import ic

# Parameters for ParticleLattice
lattice_params = {
    # include relevant parameters here
    "width": 10,
    "height": 5,
    "density": 0.3,
}


def test_compute_tm():
    flow = PoiseuilleFlow(10, 5, 1)
    lattice = ParticleLattice(10, 5)
    lattice.add_particle(5, 2, Orientation.UP)
    tm = flow.compute_tm(lattice.occupancy_map)
    print(tm[Orientation.RIGHT.value])
    ic(lattice)
    ic(lattice.occupancy_map)
    ic(flow.velocity_field)
    assert tm.shape == (4, 5, 10)
    assert tm[Orientation.UP.value].any() == 0
    assert tm[Orientation.DOWN.value].any() == 0
    assert tm[Orientation.LEFT.value].any() == 0
    assert tm[Orientation.RIGHT.value][2][5] == 1
