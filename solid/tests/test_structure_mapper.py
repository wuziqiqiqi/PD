from collections import Counter
import pytest
import numpy as np
from ase.build import bulk
from ase.neighborlist import neighbor_list
from ase import Atoms
from clease import StructureMapper


def rattled_gold_vac():
    atoms = bulk("Au", a=3.9) * (3, 3, 3)
    vac = [26, 8, 4, 0]
    for i in vac:
        del atoms[i]
    atoms.rattle(stdev=0.5)
    return atoms


def mg2sn16x6_initial():
    numbers = [
        50,
        50,
        12,
        50,
        12,
        50,
        50,
        50,
        0,
        50,
        0,
        50,
        50,
        50,
        0,
        50,
        0,
        50,
        50,
        50,
        0,
        50,
        0,
        50,
    ]
    positions = [
        [0.0, 0.0, 0.0],
        [0.0, 3.375, 3.375],
        [1.6875, 1.6875, 1.6875],
        [1.6875, 1.6875, 5.0625],
        [1.6875, 5.0625, 1.6875],
        [1.6875, 5.0625, 5.0625],
        [3.375, 0.0, 3.375],
        [3.375, 3.375, 0.0],
        [5.0625, 1.6875, 1.6875],
        [5.0625, 1.6875, 5.0625],
        [5.0625, 5.0625, 1.6875],
        [5.0625, 5.0625, 5.0625],
        [6.75, 0.0, 0.0],
        [6.75, 3.375, 3.375],
        [8.4375, 1.6875, 1.6875],
        [8.4375, 1.6875, 5.0625],
        [8.4375, 5.0625, 1.6875],
        [8.4375, 5.0625, 5.0625],
        [10.125, 0.0, 3.375],
        [10.125, 3.375, 0.0],
        [11.8125, 1.6875, 1.6875],
        [11.8125, 1.6875, 5.0625],
        [11.8125, 5.0625, 1.6875],
        [11.8125, 5.0625, 5.0625],
    ]
    cell = [[13.5, 0.0, 0.0], [0.0, 6.75, 0.0], [0.0, 0.0, 6.75]]
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=[1, 1, 1])


def mg2sn16x6_final():
    numbers = [50, 50, 12, 50, 12, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    positions = [
        [-0.73778024, 2.43e-06, 0.87741687],
        [-0.73778051, 3.33346417, 2.01317287],
        [1.68888309, 1.66673321, 1.44529559],
        [1.68888312, 1.66673289, 4.33588704],
        [1.68887889, 5.00020049, 1.44529525],
        [1.68887853, 5.00019987, 4.3358867],
        [4.11554227, -2.91e-06, 2.01317455],
        [4.11554247, 3.33346954, 0.87741795],
        [5.18664035, 1.666733, 4.33588744],
        [5.18664335, 5.00019985, 4.33588717],
        [6.95991542, 2.46e-06, 0.78537893],
        [6.9599157, 3.33346432, 2.10521272],
        [8.4444051, 1.66673259, 4.33588702],
        [8.44440496, 5.00020033, 4.3358867],
        [9.92889492, -2.76e-06, 2.10521253],
        [9.92889457, 3.33346957, 0.78537809],
        [11.70216672, 1.66673306, 4.33588657],
        [11.70216972, 5.0001999, 4.33588625],
    ]
    cell = [
        [13.51104832407987, 1.175195482e-07, 1.1506351e-09],
        [-4.9789844e-08, 6.666934187420602, -6.340775195e-07],
        [-1.843332915e-07, -9.583360596e-07, 5.781182902099645],
    ]
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=[1, 1, 1])


def mg5sn14x5_initial():
    numbers = [
        50,
        50,
        12,
        50,
        50,
        12,
        50,
        50,
        0,
        12,
        12,
        50,
        50,
        50,
        0,
        50,
        0,
        50,
        50,
        50,
        0,
        12,
        0,
        50,
    ]
    positions = [
        [0.0, 0.0, 0.0],
        [0.0, 3.375, 3.375],
        [1.6875, 1.6875, 1.6875],
        [1.6875, 1.6875, 5.0625],
        [1.6875, 5.0625, 1.6875],
        [1.6875, 5.0625, 5.0625],
        [3.375, 0.0, 3.375],
        [3.375, 3.375, 0.0],
        [5.0625, 1.6875, 1.6875],
        [5.0625, 1.6875, 5.0625],
        [5.0625, 5.0625, 1.6875],
        [5.0625, 5.0625, 5.0625],
        [6.75, 0.0, 0.0],
        [6.75, 3.375, 3.375],
        [8.4375, 1.6875, 1.6875],
        [8.4375, 1.6875, 5.0625],
        [8.4375, 5.0625, 1.6875],
        [8.4375, 5.0625, 5.0625],
        [10.125, 0.0, 3.375],
        [10.125, 3.375, 0.0],
        [11.8125, 1.6875, 1.6875],
        [11.8125, 1.6875, 5.0625],
        [11.8125, 5.0625, 1.6875],
        [11.8125, 5.0625, 5.0625],
    ]
    cell = [13.5, 6.75, 6.75]
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=[1, 1, 1])


def mg5sn14x5_final():
    numbers = [
        50,
        50,
        12,
        50,
        50,
        12,
        50,
        50,
        12,
        12,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        12,
        50,
    ]

    positions = [
        [-0.48140494, 0.0144833, 1.28457989],
        [-0.48135256, 3.18073832, 1.3010022],
        [2.68308299, 1.5976977, 1.29285962],
        [0.74928077, 1.58449916, 3.86166103],
        [2.50362203, 4.80619196, 1.30956503],
        [0.79185251, 4.79302319, 3.87840805],
        [3.85080856, -0.01562723, 3.85371676],
        [3.85078189, 3.21122682, -1.26805259],
        [6.68490289, 1.58458868, 3.86167725],
        [5.40350685, 4.80638298, 1.30957084],
        [6.7994324, 4.79310118, 3.87843199],
        [8.25224158, 0.03646678, 1.28419469],
        [8.25223131, 3.15901842, 1.30147444],
        [9.87350442, 1.58468593, 3.86177105],
        [9.95438636, 4.79320132, 3.87847087],
        [11.77915648, 0.00864269, 1.28287849],
        [11.77918874, 3.1869771, 1.30286801],
        [13.12273331, 1.58469092, 3.861781],
        [13.16078143, 4.7931881, 3.87847482],
    ]
    cell = [
        [15.54465893532526, 0.0001710673150968, 8.08448379135e-05],
        [6.84993031655e-05, 6.417021143586033, 0.0334966864736102],
        [2.0772942625e-05, -0.0262903932634979, 5.137727732454051],
    ]
    return Atoms(numbers=numbers, cell=cell, positions=positions, pbc=[1, 1, 1])


def test_rattled_structures():
    tests = [
        {
            "structure": bulk("Au", a=3.9) * (4, 4, 4),
            "expect": bulk("Au", cubic=True, a=3.9),
        },
        {
            "structure": bulk("MgO", "rocksalt", a=5.0) * (2, 2, 2),
            "expect": bulk("MgO", "rocksalt", a=5.0, cubic=True),
        },
    ]

    mapper = StructureMapper()
    for test in tests:
        rattled = test["structure"].copy()
        rattled.rattle(stdev=0.01, seed=0)
        recovered = mapper.refine(rattled)

        pos = np.sort(recovered.get_positions().ravel())
        pos_expect = np.sort(test["expect"].get_positions().ravel())
        assert np.allclose(pos, pos_expect)


def test_vacancies_no_distortion():
    atoms = bulk("Au", a=3.9) * (3, 3, 3)
    atoms[0].symbol = "X"
    atoms[10].symbol = "X"

    no_vac = atoms.copy()
    del atoms[10]
    del atoms[0]

    mapper = StructureMapper(symprec=0.1)
    recovered = mapper.refine(no_vac)

    # Vacancy to Au ratio
    count = Counter(recovered.numbers)
    x_to_au = count[0] / count[79]
    assert x_to_au == pytest.approx(2.0 / 25.0)

    at_index, dist = neighbor_list("id", recovered, 3.0)
    coordination = np.bincount(at_index)
    assert np.all(coordination == 12)
    assert np.allclose(dist, 3.9 / np.sqrt(2.0))


def test_vacancies_with_distortion():
    tests = [
        {
            "structure": rattled_gold_vac(),
            "template": bulk("Au", a=3.7) * (3, 3, 3),
            "expect_vac": [0, 4, 8, 26],
        },
        {
            "structure": mg2sn16x6_final(),
            "template": mg2sn16x6_initial(),
            "expect_vac": [8, 10, 14, 16, 20, 22],
        },
        {
            "structure": mg5sn14x5_final(),
            "template": mg5sn14x5_initial(),
            "expect_vac": [8, 14, 16, 20, 22],
        },
    ]

    mapper = StructureMapper()
    for test in tests:
        recovered, _ = mapper.snap_to_lattice(test["structure"], test["template"])
        for i in test["expect_vac"]:
            assert recovered[i].symbol == "X"
