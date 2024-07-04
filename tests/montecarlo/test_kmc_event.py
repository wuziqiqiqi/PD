from cleases.montecarlo import NeighbourSwap
from ase.build import bulk


def test_neighbour_swap():
    atoms = bulk("Al", a=4.05) * (3, 3, 3)

    # Set cutoff to only nearest neighbours
    swapper = NeighbourSwap(atoms, 3.0)
    for n in swapper.nl:
        assert len(n) == 12
