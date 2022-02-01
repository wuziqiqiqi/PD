import pytest

import numpy as np
from ase.build import bulk

from clease.cluster import ClusterGenerator
from clease.datastructures import FourVector, Figure


def test_sites_cutoff_fcc():
    atoms = bulk("Al", a=4.05)
    generator = ClusterGenerator(atoms)
    indices = generator.sites_within_cutoff(3.0, FourVector(0, 0, 0, 0))
    indices = list(indices)

    # For FCC there should be 12 sites within the cutoff
    assert len(list(indices)) == 12

    # FCC within 4.1
    indices = list(generator.sites_within_cutoff(4.1, FourVector(0, 0, 0, 0)))
    assert len(indices) == 18

    # FCC within 5.0
    indices = list(generator.sites_within_cutoff(5.0, FourVector(0, 0, 0, 0)))
    assert len(indices) == 42


def test_sites_cutoff_bcc():
    a = 3.8
    atoms = bulk("Fe", a=a)
    generator = ClusterGenerator(atoms)

    # Neighbour distances
    nn = np.sqrt(3) * a / 2.0
    snn = a
    indices = list(generator.sites_within_cutoff(nn + 0.01, FourVector(0, 0, 0, 0)))
    assert len(indices) == 8
    indices = list(generator.sites_within_cutoff(snn + 0.01, FourVector(0, 0, 0, 0)))
    assert len(indices) == 14


def test_generate_pairs_fcc():
    atoms = bulk("Al", a=4.05)
    generator = ClusterGenerator(atoms)
    clusters, fps = generator.generate(2, 5.0, 0)
    assert len(clusters) == 3
    assert len(fps) == 3


def test_equivalent_sites():
    atoms = bulk("Au", a=3.8)
    generator = ClusterGenerator(atoms)

    # Test pairs
    clusters, fps = generator.generate(2, 6.0, 0)
    for c in clusters:
        equiv = generator.equivalent_sites(c[0])
        assert equiv == [[0, 1]]

    # Test a triplet
    clusters, fps = generator.generate(3, 3.5, 0)

    # For the smalles triplet all sites should be equivalent
    equiv = generator.equivalent_sites(clusters[0][0])
    assert equiv == [[0, 1, 2]]


def test_get_lattice():
    tests = [
        {"prim": bulk("Al"), "atoms": bulk("Al") * (2, 2, 2), "site": 4, "lattice": 0},
        {
            "prim": bulk("LiX", "rocksalt", 4.0),
            "atoms": bulk("LiX", "rocksalt", 4.0) * (1, 2, 3),
            "site": 4,
            "lattice": 0,
        },
        {
            "prim": bulk("LiX", "rocksalt", 4.0),
            "atoms": bulk("LiX", "rocksalt", 4.0) * (1, 2, 3),
            "site": 5,
            "lattice": 1,
        },
    ]

    for i, test in enumerate(tests):
        test["atoms"].wrap()
        test["prim"].wrap()
        for at in test["prim"]:
            at.tag = at.index
        pos = test["atoms"][test["site"]].position
        gen = ClusterGenerator(test["prim"])
        lattice = gen.get_lattice(pos)
        assert lattice == test["lattice"]


@pytest.mark.parametrize(
    "test",
    [
        {
            "input": Figure([FourVector(1, 2, 3, 0), FourVector(3, 4, 1, 0)]),
            "result": 3.4641,
        },
        {
            "input": Figure([FourVector(1, 2, 3, 0), FourVector(2, 3, 4, 0)]),
            "result": 1.7321,
        },
        {
            "input": Figure(
                [FourVector(1, 2, 3, 0), FourVector(2, 3, 4, 0), FourVector(3, 4, 1, 0)]
            ),
            "result": 3.4641,
        },
    ],
)
def test_get_max_distance(test):
    a = 3.8
    atoms = bulk("Fe", a=a)
    atoms.cell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    generator = ClusterGenerator(atoms)
    assert pytest.approx(test["result"]) == round(generator.get_max_distance(test["input"]), 4)


def test_cutoff():
    """Rocksalt used to find clusters which were slightly larger than the diameter.
    Test that all diameters are less-than or equal to the maximum expected diameter"""
    atoms = bulk("NaCl", "rocksalt", a=5.0)
    gen = ClusterGenerator(atoms)
    max_diameter = 5.0
    for size in [2, 3, 4, 5]:
        for ref_lattice in range(len(atoms)):
            clusters, _ = gen.generate(size, max_diameter, ref_lattice)
            for group in clusters:
                # Also test that all diameters are equal to the same diameter
                # in the same group
                fig0_dia = group[0].get_diameter(atoms)
                for figure in group:
                    assert isinstance(figure, Figure)
                    fig_dia = figure.get_diameter(atoms)
                    assert fig_dia <= max_diameter
                    assert fig_dia == pytest.approx(fig0_dia)
