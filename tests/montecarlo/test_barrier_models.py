import numpy as np
from cleases.montecarlo import BEPBarrier
from cleases.settings import CEBulk, Concentration
from cleases.calculator import attach_calculator
from cleases.datastructures import SystemChange


def test_loc_environ_barrier(db_name):
    conc = Concentration(basis_elements=[["Au", "Cu", "X"]])
    settings = CEBulk(
        conc,
        crystalstructure="fcc",
        size=[1, 1, 1],
        max_cluster_dia=[3.0],
        db_name=db_name,
    )

    eci = {"c0": 0.0, "c1_0": 0.0, "c2_d0000_0_00": 0.0}

    atoms = settings.atoms.copy() * (4, 4, 4)
    atoms = attach_calculator(settings, atoms, eci)

    dilute_barriers = {"Au": 0.5, "Cu": 0.4}

    barrier = BEPBarrier(dilute_barriers)

    # Insert some Cu
    for i in range(4):
        atoms[i].symbol = "Cu"
    atoms[10].symbol = "X"
    atoms.get_potential_energy()  # Trigger a calculation
    atoms.calc.clear_history()

    init_numbers = atoms.numbers.copy()

    # Perform a swap
    other = 11
    s = atoms[other].symbol
    barrier(atoms, [SystemChange(10, "X", s, ""), SystemChange(other, s, "X", "")])

    # Make sure that the atoms object remains unchanges by the barrier function
    assert np.all(init_numbers == atoms.numbers)
