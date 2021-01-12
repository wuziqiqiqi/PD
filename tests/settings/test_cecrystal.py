"""Test to initiatialize CE using a CECrystal."""
import os
import json
from pathlib import Path
import numpy as np
import pytest
from ase.db import connect
from ase.spacegroup import crystal
from clease.settings import CECrystal, settings_from_json, Concentration
from clease.corr_func import CorrFunction
from clease import NewStructures
from clease.structgen import MaxAttemptReachedError
from clease.tools import wrap_and_sort_by_position

# If this is True, the JSON file containing the correlation functions
# Used to check consistency of the reference functions is updated
# This should normally be False
update_reference_file = False
tol = 1E-9
ref_file = Path(__file__).parent / 'reference_corr_funcs_crystal.json'


def update_cf(new_cf):
    global ref_file
    with ref_file.open('w') as file:
        json.dump(new_cf, file, indent=2, separators=(',', ': '))


@pytest.fixture
def all_cf():
    global ref_file
    if not ref_file.exists():
        update_cf({})
    with ref_file.open() as file:
        return json.load(file)


@pytest.fixture
def check_cf(all_cf):
    global update_reference_file

    def _check_cf(cf, cf_key):
        if update_reference_file:
            all_cf[cf_key] = cf
        for key in cf.keys():
            assert cf[key] == pytest.approx(all_cf[cf_key][key])
        if update_reference_file:
            update_cf(all_cf)

    return _check_cf


def get_figures_of_family(settings, cname):
    """Return the figures of a given cluster family."""
    figures = []
    clusters = settings.cluster_list.get_by_name(cname)
    for cluster in clusters:
        figures.append(cluster.indices)
    return figures


def test_spgroup_217(db_name, tmpdir, all_cf):
    """Test the initialization of spacegroup 217."""
    a = 10.553
    b = 10.553
    c = 10.553
    alpha = 90
    beta = 90
    gamma = 90
    cellpar = [a, b, c, alpha, beta, gamma]
    basis = [[0, 0, 0], [0.324, 0.324, 0.324], [0.3582, 0.3582, 0.0393], [0.0954, 0.0954, 0.2725]]
    basis_elements = [["Al", "Mg"], ["Al", "Mg"], ["Al", "Mg"], ["Al", "Mg"]]

    # Test with grouped basis
    conc = Concentration(basis_elements=basis_elements, grouped_basis=[[0, 1, 2, 3]])
    bsg = CECrystal(concentration=conc,
                    spacegroup=217,
                    basis=basis,
                    cellpar=cellpar,
                    max_cluster_size=3,
                    db_name=db_name,
                    size=[1, 1, 1],
                    max_cluster_dia=[3.5, 3.5])
    bsg.include_background_atoms = True
    bsg.skew_threshold = 80

    # The correlation functions are actually calculated for the
    # conventional cell
    atoms = crystal(symbols=['Al', 'Al', 'Al', 'Al'],
                    cellpar=cellpar,
                    spacegroup=217,
                    primitive_cell=False,
                    basis=basis)
    atoms = wrap_and_sort_by_position(atoms)
    # atoms = bsg.atoms.copy()

    atoms.symbols[[0, 10, 20, 30]] = "Mg"
    corr = CorrFunction(bsg)
    cf = corr.get_cf(atoms)

    if update_reference_file:
        all_cf["sp_217_grouped"] = cf
    for key in cf.keys():
        assert cf[key] == pytest.approx(all_cf["sp_217_grouped"][key])

    os.remove(db_name)

    bsg.basis_func_type = 'binary_linear'
    fname = str(tmpdir / 'demo_save.json')
    bsg.save(fname)
    bsg_loaded = settings_from_json(fname)
    for k, v in bsg.__dict__.items():
        if k in ['kwargs', 'size', 'template_atoms', 'atoms_mng', 'trans_matrix', 'cluster_list']:
            # Skip attributes not expected to be equal after load/save
            continue
        if isinstance(v, np.ndarray):
            assert np.allclose(v, bsg_loaded.__dict__[k])
        else:
            assert v == bsg_loaded.__dict__[k]
    assert bsg.skew_threshold == bsg_loaded.skew_threshold

    if update_reference_file:
        update_cf(all_cf)


def test_two_grouped_basis(db_name, check_cf):
    # ---------------------------------- #
    # 2 grouped_basis                    #
    # ---------------------------------- #
    basis_elements = [['Li', 'X', 'V'], ['Li', 'X', 'V'], ['O', 'F']]
    grouped_basis = [[0, 1], [2]]
    conc = Concentration(basis_elements=basis_elements, grouped_basis=grouped_basis)

    cellpar = [5.123, 5.123, 13.005, 90., 90., 120.]
    basis = [(0.00, 0.00, 0.00), (1. / 3, 2. / 3, 0.00), (1. / 3, 0.00, 0.25)]
    bsg = CECrystal(concentration=conc,
                    spacegroup=167,
                    basis=basis,
                    cellpar=cellpar,
                    size=[1, 1, 1],
                    db_name=db_name,
                    max_cluster_size=3,
                    max_cluster_dia=[2.5, 2.5])
    assert bsg.unique_elements == ['F', 'Li', 'O', 'V', 'X']
    assert bsg.spin_dict == {'F': 2.0, 'Li': -2.0, 'O': 1.0, 'V': -1.0, 'X': 0}

    assert len(bsg.basis_functions) == 4
    assert bsg.num_basis == 2
    assert len(bsg.index_by_basis) == 2

    # The correlation functions in the reference file is calculated for
    # the conventional cell
    atoms = crystal(symbols=['Li', 'Li', 'O'],
                    basis=basis,
                    cellpar=cellpar,
                    primitive_cell=False,
                    spacegroup=167)
    atoms = wrap_and_sort_by_position(atoms)
    indx_to_X = [6, 33, 8, 35]
    for indx in indx_to_X:
        atoms[indx].symbol = "X"
    corr = CorrFunction(bsg)
    cf = corr.get_cf(atoms)
    cf_key = "Li_X_V_O_F"

    check_cf(cf, cf_key)


def test_two_grouped_basis_probe_structure(db_name, check_cf):
    # ---------------------------------- #
    # 2 grouped_basis                    #
    # ---------------------------------- #
    # initial_pool + probe_structures    #
    # ---------------------------------- #
    basis_elements = [['O', 'X'], ['O', 'X'], ['O', 'X'], ['Ta']]
    grouped_basis = [[0, 1, 2], [3]]
    conc = Concentration(basis_elements=basis_elements, grouped_basis=grouped_basis)

    bsg = CECrystal(concentration=conc,
                    spacegroup=55,
                    basis=[(0., 0., 0.), (0.3894, 0.1405, 0.), (0.201, 0.3461, 0.5),
                           (0.2244, 0.3821, 0.)],
                    cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                    size=[1, 2, 2],
                    db_name=db_name,
                    max_cluster_size=3,
                    max_cluster_dia=[3.0, 3.0])
    bsg.include_background_atoms = True

    assert bsg.unique_elements == ['O', 'Ta', 'X']
    assert bsg.spin_dict == {'O': 1.0, 'Ta': -1.0, 'X': 0.0}
    assert len(bsg.basis_functions) == 2
    assert bsg.num_basis == 2
    assert len(bsg.index_by_basis) == 2

    atoms = bsg.atoms.copy()
    indx_to_X = [0, 4, 8, 12, 16]
    for indx in indx_to_X:
        atoms[indx].symbol = "X"
    corr = CorrFunction(bsg)
    cf = corr.get_cf(atoms)
    cf_key = "Ta_O_X_grouped"
    check_cf(cf, cf_key)

    try:
        ns = NewStructures(settings=bsg, struct_per_gen=2)
        ns.generate_initial_pool()
        ns = NewStructures(settings=bsg, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0,
                                    final_temp=0.001,
                                    num_temp=5,
                                    num_steps_per_temp=100,
                                    approx_mean_var=True)

        db = connect(db_name)
        for row in db.select(struct_type='initial'):
            atoms = row.toatoms()
            cf = corr.get_cf(atoms)
            for key, value in cf.items():
                assert value == pytest.approx(row["polynomial_cf"][key])

    except MaxAttemptReachedError as exc:
        print(str(exc))


def test_two_grouped_basis_background_atoms_probe_structure(db_name, check_cf):
    # ---------------------------------- #
    # 2 grouped_basis + background atoms #
    # ---------------------------------- #
    # initial_pool + probe_structures    #
    # ---------------------------------- #
    basis_elements = [['O', 'X'], ['Ta'], ['O', 'X'], ['O', 'X']]
    grouped_basis = [[1], [0, 2, 3]]
    conc = Concentration(basis_elements=basis_elements, grouped_basis=grouped_basis)

    bsg = CECrystal(concentration=conc,
                    spacegroup=55,
                    basis=[(0., 0., 0.), (0.2244, 0.3821, 0.), (0.3894, 0.1405, 0.),
                           (0.201, 0.3461, 0.5)],
                    cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                    size=[2, 2, 3],
                    db_name=db_name,
                    max_cluster_size=3,
                    max_cluster_dia=[3.0, 3.0])

    assert bsg.unique_elements == ['O', 'Ta', 'X']
    assert bsg.spin_dict == {'O': 1.0, 'X': -1.0}
    assert bsg.basis_elements == [['Ta'], ['O', 'X']]
    assert len(bsg.basis_functions) == 1
    assert bsg.num_basis == 2
    assert len(bsg.index_by_basis) == 2

    try:
        ns = NewStructures(settings=bsg, struct_per_gen=2)
        ns.generate_initial_pool()
        ns = NewStructures(settings=bsg, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0,
                                    final_temp=0.001,
                                    num_temp=5,
                                    num_steps_per_temp=100,
                                    approx_mean_var=True)
        atoms = bsg.atoms.copy()
        indx_to_X = [0, 4, 8, 12, 16]
        for indx in indx_to_X:
            atoms[indx].symbol = "X"
        corr = CorrFunction(bsg)
        cf = corr.get_cf(atoms)
        cf_key = "Ta_O_X_ungrouped"
        check_cf(cf, cf_key)

        db = connect(db_name)
        for row in db.select(struct_type='initial'):
            atoms = row.toatoms()
            cf = corr.get_cf(atoms)
            for key, value in cf.items():
                assert value == pytest.approx(row["polynomial_cf"][key])

    except MaxAttemptReachedError as exc:
        print(str(exc))


def test_narrow_angle_crystal(db_name):
    """Test that Probestructure works for crystals with narrow angles.

    This test a crystal with internal angles 50, 20, 15 degree.
    """
    basis_elements = [['Mg', 'Si']]
    conc = Concentration(basis_elements=basis_elements)
    bsg = CECrystal(concentration=conc,
                    spacegroup=225,
                    basis=[(0.0, 0.0, 0.0)],
                    cellpar=[4.0, 4.0, 4.0, 50.0, 40.0, 15.0],
                    db_name=db_name,
                    size=[2, 2, 1],
                    max_cluster_size=3,
                    max_cluster_dia=[1.05, 1.05])
    bsg.skew_threshold = 10000

    assert len(bsg.index_by_sublattice) == 1

    try:
        ns = NewStructures(settings=bsg, struct_per_gen=2)
        ns.generate_initial_pool()
        ns = NewStructures(settings=bsg, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0,
                                    final_temp=0.001,
                                    num_temp=5,
                                    num_steps_per_temp=100,
                                    approx_mean_var=True)
    except MaxAttemptReachedError as exc:
        print(str(exc))


def test_bkg_symb_in_additional_basis(db_name):
    conc = Concentration(basis_elements=[['Mg', 'Sn', 'X'], ['Sn']])
    settings = CECrystal(cellpar=[6.75, 6.75, 6.75, 90, 90, 90],
                         basis=[(0.25, 0.25, 0.25), (0, 0, 0)],
                         concentration=conc,
                         spacegroup=225,
                         size=[(-1, 1, 1), (1, -1, 1), (1, 1, -1)],
                         supercell_factor=1,
                         db_name=db_name,
                         max_cluster_size=2,
                         max_cluster_dia=5.0)
    bfs = settings.basis_functions
    assert len(bfs) == 2

    for bf in bfs:
        keys = sorted(list(bf.keys()))
        assert keys == ['Mg', 'Sn', 'X']
