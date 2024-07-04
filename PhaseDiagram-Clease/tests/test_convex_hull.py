import pytest
from clease import ConvexHull
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.build import bulk
import numpy as np
import os


@pytest.fixture
def rng():
    return np.random.default_rng(28)


def test_binary(db_name):
    # from matplotlib import pyplot as plt
    # plt.switch_backend("agg")

    db = connect(db_name)

    # Create energies that we know are on the
    # convex hull
    cnv_hull_enegies = [-x * (8 - x) - x + 0.2 for x in range(9)]

    for n_cu in range(9):
        atoms = bulk("Au")
        atoms = atoms * (2, 2, 2)

        for i in range(n_cu):
            atoms[i].symbol = "Cu"

        calc = SinglePointCalculator(atoms, energy=cnv_hull_enegies[n_cu])
        atoms.calc = calc
        db.write(atoms, converged=True)

        # Create a new structure with exactly the same
        # composition, but higher energy
        calc = SinglePointCalculator(atoms, energy=cnv_hull_enegies[n_cu] + 0.5)
        atoms.calc = calc
        db.write(atoms, converged=True)

    cnv_hull = ConvexHull(db_name, conc_ranges={"Au": (0, 1)})
    energies = []
    comp = []
    for row in db.select():
        energies.append(row.energy / row.natoms)
        count = row.count_atoms()

        for k in ["Au", "Cu"]:
            if k in count.keys():
                count[k] /= row.natoms
            else:
                count[k] = 0.0
        comp.append(count)
    os.remove(db_name)

    # Calculate distance to the convex hull
    for c, tot_en in zip(comp, energies):
        cnv_hull.cosine_similarity_convex_hull(c, tot_en)

    # fig = cnv_hull.plot()
    # assert len(fig.get_axes()) == 1


def test_syst_with_one_fixed_comp(db_name):
    db = connect(db_name)

    # Create energies that we know are on the
    # convex hull

    for n_cu in range(6):
        atoms = bulk("Au")
        atoms = atoms * (2, 2, 2)

        atoms[0].symbol = "Zn"
        atoms[1].symbol = "Zn"

        for i in range(n_cu):
            atoms[i + 2].symbol = "Cu"

        calc = SinglePointCalculator(atoms, energy=np.random.rand())
        atoms.calc = calc
        db.write(atoms, converged=True)

    cnv_hull = ConvexHull(db_name, conc_ranges={"Au": (0, 1)})

    os.remove(db_name)
    assert cnv_hull._unique_elem == ["Au", "Cu", "Zn"]
    # fig = cnv_hull.plot()
    # assert len(fig.get_axes()) == 1


def test_system_multiple_endpoints(db_name, rng):
    db = connect(db_name)
    atoms_orig = bulk("Au") * (1, 1, 2)

    # Create some atoms objects at each extremum,
    # with vastly different energies
    energies = -350 * rng.random(25)
    mult = {"Au": 1, "Zn": 2}
    for en in energies:
        for sym, m in mult.items():
            atoms = atoms_orig.copy()
            atoms.symbols = sym
            calc = SinglePointCalculator(atoms, energy=en * m)
            atoms.calc = calc
            db.write(atoms, converged=True)

    cnv_hull = ConvexHull(db_name, conc_ranges={"Au": (0, 1)})

    end_points = cnv_hull.end_points
    # Ensure the end-points correspond to the energy of the minimum energy structure
    for sym, m in mult.items():
        assert end_points[sym]["energy"] == pytest.approx(energies.min() * m / len(atoms_orig))
