import pytest
import numpy as np

from clease.calculator import attach_calculator
from clease.montecarlo import SGCMonteCarlo
from clease.montecarlo.observers import SGCState, MultiStateSGCConcObserver
from clease.settings import CEBulk, Concentration
from clease.corr_func import CorrFunction
from clease.montecarlo.constraints import ConstrainElementInserts
from clease.montecarlo.constraints import PairConstraint
from clease.montecarlo import RandomFlip
from clease.tools import species_chempot2eci


def test_run(db_name):
    conc = Concentration(basis_elements=[['Au', 'Cu']])
    settings = CEBulk(db_name=db_name,
                      concentration=conc,
                      crystalstructure='fcc',
                      a=4.0,
                      max_cluster_dia=[5.0, 4.1],
                      size=[2, 2, 2])

    atoms = settings.atoms.copy() * (3, 3, 3)
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}

    eci['c0'] = -1.0
    eci['c2_d0000_0_00'] = -0.2
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)

    E = []
    for T in [5000, 2000, 1000, 500, 100]:
        mc = SGCMonteCarlo(atoms, T, symbols=['Au', 'Cu'])
        mc.run(steps=10000, chem_pot={'c1_0': -0.02})
        E.append(mc.get_thermodynamic_quantities()['energy'])

    cf_calc = atoms.calc.get_cf()
    cf_scratch = cf.get_cf(atoms)

    for k, v in cf_calc.items():
        assert v == pytest.approx(cf_calc[k])

    # Make sure that the energies are decreasing
    for i in range(1, len(E)):
        assert E[i - 1] >= E[i]


def test_constrain_inserts(db_name):
    conc = Concentration(basis_elements=[['Si', 'X'], ['O', 'C']])
    settings = CEBulk(db_name=db_name,
                      concentration=conc,
                      crystalstructure='rocksalt',
                      a=4.0,
                      max_cluster_dia=[2.51, 3.0],
                      size=[2, 2, 2])
    atoms = settings.atoms.copy() * (3, 3, 3)
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}

    eci['c0'] = -1.0
    eci['c2_d0000_0_00'] = -0.2
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)

    generator = RandomFlip(['Si', 'X', 'O', 'C'], atoms)
    elem_basis = [['Si', 'X'], ['O', 'C']]
    cnst = ConstrainElementInserts(atoms, settings.index_by_basis, elem_basis)
    generator.add_constraint(cnst)
    mc = SGCMonteCarlo(atoms, 100000, generator=generator)

    chem_pot = {'c1_0': 0.0, 'c1_1': -0.1, 'c1_2': 0.1}
    orig_symbols = [a.symbol for a in atoms]
    mc.run(steps=1000, chem_pot=chem_pot)

    new_symb = [a.symbol for a in atoms]
    # Confirm that swaps have taken place
    assert not all(map(lambda x: x[0] == x[1], zip(orig_symbols, new_symb)))

    # Check that we only have correct entries
    for allowed, basis in zip(elem_basis, settings.index_by_basis):
        for index in basis:
            assert atoms[index].symbol in allowed


def test_pair_constraint(db_name):
    a = 4.0
    conc = Concentration(basis_elements=[['Si', 'X']])
    settings = CEBulk(db_name=db_name,
                      concentration=conc,
                      crystalstructure='fcc',
                      a=a,
                      max_cluster_dia=[3.9, 3.0],
                      size=[2, 2, 2])

    atoms = settings.atoms.copy() * (3, 3, 3)
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)

    generator = RandomFlip(['Si', 'X'], atoms)
    cluster = settings.cluster_list.get_by_name("c2_d0000_0")[0]
    cnst = PairConstraint(['X', 'X'], cluster, settings.trans_matrix, atoms)
    generator.add_constraint(cnst)

    mc = SGCMonteCarlo(atoms, 100000000, generator=generator)

    nn_dist = a / np.sqrt(2.0)
    for _ in range(20):
        mc.run(10, chem_pot={'c1_0': 0.0})

        X_idx = [atom.index for atom in atoms if atom.symbol == 'X']

        if len(X_idx) <= 1:
            continue

        # Check that there are no X that are nearest neighbours
        dists = [atoms.get_distance(i1, i2) for i1 in X_idx for i2 in X_idx[i1 + 1:]]
        assert all(d > 1.05 * nn_dist for d in dists)


def test_multi_state_sgc_obs(db_name):
    conc = Concentration([['Ag', 'Pt']])
    settings = CEBulk(conc,
                      a=4.0,
                      crystalstructure='sc',
                      db_name=db_name,
                      max_cluster_dia=[4.01],
                      size=[4, 4, 4])
    atoms = settings.atoms.copy()
    eci = {'c0': 0.0, 'c1_0': 0.0, 'c2_d0000_0_00': 0.1}

    atoms = attach_calculator(settings, atoms=atoms, eci=eci)
    T = 400.0
    sgc = SGCMonteCarlo(atoms, T, symbols=['Ag', 'Pt'])

    chem_pots = [{
        'Ag': 0.0,
    }, {
        'Ag': 0.5,
    }, {
        'Ag': -0.5,
    }]

    chem_pots = [species_chempot2eci(settings.basis_functions, c) for c in chem_pots]
    ref_pot = chem_pots[0]

    states = [SGCState(T, c) for c in chem_pots]
    observer = MultiStateSGCConcObserver(
        SGCState(T, ref_pot),
        states,
        atoms.calc,
    )

    # Seed python's random number generator and numpy's
    np.random.seed(0)

    sgc.attach(observer)
    sgc.run(steps=2000, chem_pot=ref_pot)

    # Check consistency. By construction the Ag concentration varies in the same
    # direction as the chemical potential (high potential --> more Ag)
    avg = observer.get_averages()
    conc_center = avg['400K_c1_0plus0_singlet_c1_0']
    conc_plus = avg['400K_c1_0plus500_singlet_c1_0']
    conc_minus = avg['400K_c1_0minus500_singlet_c1_0']

    assert conc_plus > conc_center
    assert conc_minus < conc_center
