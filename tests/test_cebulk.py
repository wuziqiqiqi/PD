"""Test to initiatialize CE using a CEBulk.

1. Initialize the CE
2. Add a few structures
3. Compute the energy
4. Run the evaluation routine
"""

import os
import json
import pytest
from pathlib import Path

import numpy as np
from ase.calculators.emt import EMT
from ase.db import connect
from ase.build import bulk
from ase.build import make_supercell

from clease.settings import CEBulk, settings_from_json, Concentration
from clease import NewStructures, Evaluate
from clease.corr_func import CorrFunction
from clease.new_struct import MaxAttemptReachedError
from clease.tools import update_db
from clease.basis_function import (Polynomial, Trigonometric, BinaryLinear)

# If this is True, the JSON file containing the correlation functions
# Used to check consistency of the reference functions is updated
# This should normally be False
update_reference_file = False
tol = 1E-9
ref_file = Path(__file__).parent / 'reference_corr_funcs_bulk.json'


def update_cf(new_cf):
    with ref_file.open('w') as file:
        json.dump(new_cf, file, indent=2, separators=(',', ': '))


@pytest.fixture
def all_cf():
    if not ref_file.exists():
        update_cf({})
    with ref_file.open() as file:
        return json.load(file)


def get_figures_of_family(settings, cname):
    """Return the figures of a given cluster family."""
    figures = []
    clusters = settings.cluster_list.get_by_name(cname)
    for cluster in clusters:
        figures.append(cluster.indices)
    return figures


def calculate_cf(settings, atoms):
    cf = CorrFunction(settings)
    cf_dict = cf.get_cf(atoms)
    return cf_dict


def test_load_from_db(db_name):
    basis_elements = [['Au', 'Cu']]
    conc = Concentration(basis_elements=basis_elements)
    settings = CEBulk(concentration=conc,
                      crystalstructure='fcc',
                      a=4.05,
                      size=[1, 1, 1],
                      db_name=db_name,
                      max_cluster_dia=[4.3, 4.3, 4.3],
                      max_cluster_size=4)
    orig_atoms = settings.atoms.copy()
    atoms = bulk('Au', crystalstructure='fcc', a=4.05, cubic=True)
    settings.set_active_template(atoms=atoms)

    # Try to read back the old atoms
    settings.set_active_template(atoms=orig_atoms)


def test_max_cluster_dia():
    basis_elements = [['Au', 'Cu']]
    conc = Concentration(basis_elements=basis_elements)
    mcd = [4.3, 4.3, 4.3]
    mcd_orig = mcd.copy()
    settings = CEBulk(concentration=conc,
                      crystalstructure='fcc',
                      a=4.05,
                      size=[1, 1, 1],
                      max_cluster_dia=mcd,
                      max_cluster_size=4)

    # Ensure max cluster dia has not been mutated
    assert not mcd is mcd_orig
    assert mcd == mcd_orig

    # Explicitly test format_max_cluster dia, ensure no mutation still
    out = settings._format_max_cluster_dia(mcd)
    assert mcd == mcd_orig
    assert isinstance(out, np.ndarray)
    assert type(mcd) != type(out)
    assert type(settings.max_cluster_dia) == type(out)
    assert settings.max_cluster_dia.tolist() == out.tolist()


def test_corrfunc(db_name, all_cf):
    basis_elements = [['Au', 'Cu']]
    conc = Concentration(basis_elements=basis_elements)
    settings = CEBulk(concentration=conc,
                      crystalstructure='fcc',
                      a=4.05,
                      size=[3, 3, 3],
                      db_name=db_name,
                      max_cluster_dia=[4.3, 4.3, 4.3],
                      max_cluster_size=4)
    atoms = settings.atoms.copy()
    atoms[0].symbol = 'Cu'
    atoms[3].symbol = 'Cu'
    cf = calculate_cf(settings, atoms)

    if update_reference_file:
        all_cf["binary_fcc"] = cf
    for key in cf.keys():
        assert cf[key] == pytest.approx(all_cf["binary_fcc"][key])

    os.remove(db_name)

    basis_elements = [['Li', 'V'], ['X', 'O']]
    conc = Concentration(basis_elements=basis_elements)
    settings = CEBulk(concentration=conc,
                      crystalstructure="rocksalt",
                      a=4.0,
                      size=[2, 2, 1],
                      db_name=db_name,
                      max_cluster_size=3,
                      max_cluster_dia=[4.01, 4.01])
    atoms = settings.atoms.copy()
    Li_ind = [atom.index for atom in atoms if atom.symbol == 'Li']
    X_ind = [atom.index for atom in atoms if atom.symbol == 'X']
    atoms[Li_ind[0]].symbol = 'V'
    atoms[X_ind[0]].symbol = 'O'
    cf = calculate_cf(settings, atoms)
    if update_reference_file:
        all_cf["two_basis"] = cf
    for key in cf.keys():
        assert cf[key] == pytest.approx(all_cf["two_basis"][key])
    os.remove(db_name)

    basis_elements = [['Na', 'Cl'], ['Na', 'Cl']]
    conc = Concentration(basis_elements=basis_elements, grouped_basis=[[0, 1]])
    settings = CEBulk(concentration=conc,
                      crystalstructure="rocksalt",
                      a=4.0,
                      size=[2, 2, 1],
                      db_name=db_name,
                      max_cluster_size=3,
                      max_cluster_dia=[4.01, 4.01])
    atoms = settings.atoms.copy()
    atoms[1].symbol = 'Cl'
    atoms[7].symbol = 'Cl'
    cf = calculate_cf(settings, atoms)
    if update_reference_file:
        all_cf["one_grouped_basis"] = cf
    for key in cf.keys():
        assert cf[key] == pytest.approx(all_cf["one_grouped_basis"][key])
    os.remove(db_name)

    basis_elements = [['Ca'], ['O', 'F'], ['O', 'F']]
    conc = Concentration(basis_elements=basis_elements, grouped_basis=[[0], [1, 2]])
    settings = CEBulk(concentration=conc,
                      crystalstructure="fluorite",
                      a=4.0,
                      size=[2, 2, 2],
                      db_name=db_name,
                      max_cluster_size=3,
                      max_cluster_dia=[4.01, 4.01])
    atoms = settings.atoms.copy()
    O_ind = [atom.index for atom in atoms if atom.symbol == 'O']
    atoms[O_ind[0]].symbol = 'F'
    atoms[O_ind[1]].symbol = 'F'
    cf = calculate_cf(settings, atoms)
    if update_reference_file:
        all_cf["two_grouped_basis_bckgrnd"] = cf

    for key in cf.keys():
        assert cf[key] == pytest.approx(all_cf["two_grouped_basis_bckgrnd"][key])

    if update_reference_file:
        update_cf(all_cf)


def test_binary_system(db_name, tmpdir):
    """Verifies that one can run a CE for the binary Au-Cu system.

    The EMT calculator is used for energy calculations
    """
    basis_elements = [['Au', 'Cu']]
    conc = Concentration(basis_elements=basis_elements)
    bc_settings = CEBulk(concentration=conc,
                         crystalstructure='fcc',
                         a=4.05,
                         size=[3, 3, 3],
                         db_name=db_name)

    newstruct = NewStructures(bc_settings, struct_per_gen=3)
    newstruct.generate_initial_pool()

    # Compute the energy of the structures
    calc = EMT()
    database = connect(db_name)

    # Write the atoms to the database
    # for atoms, kvp in zip(all_atoms, key_value_pairs):
    for row in database.select([("converged", "=", False)]):
        atoms = row.toatoms()
        atoms.calc = calc
        atoms.get_potential_energy()
        update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)
    # Evaluate
    evaluator = Evaluate(bc_settings, fitting_scheme="l2", alpha=1E-6)

    # Test subclusters for pairs
    for cluster in bc_settings.cluster_list.get_by_size(2):
        sub_cl = bc_settings.cluster_list.get_subclusters(cluster)
        sub_cl_name = set([c.name for c in sub_cl])
        assert sub_cl_name == set(["c0", "c1"])

    # Test a few known clusters. Triplet nearest neighbour
    name = "c3_d0000_0"
    triplet = bc_settings.cluster_list.get_by_name(name)[0]
    sub_cl = bc_settings.cluster_list.get_subclusters(triplet)
    sub_cl_name = set([c.name for c in sub_cl])
    assert sub_cl_name == set(["c0", "c1", "c2_d0000_0"])

    name = "c3_d0001_0"
    triplet = bc_settings.cluster_list.get_by_name(name)[0]
    sub_cl = (bc_settings.cluster_list.get_subclusters(triplet))
    sub_cl_name = set([c.name for c in sub_cl])
    assert sub_cl_name == set(["c0", "c1", "c2_d0000_0", "c2_d0001_0"])

    name = "c4_d0000_0"
    quad = bc_settings.cluster_list.get_by_name(name)[0]
    sub_cl = bc_settings.cluster_list.get_subclusters(quad)
    sub_cl_name = set([c.name for c in sub_cl])
    assert sub_cl_name == set(["c0", "c1", "c2_d0000_0", "c3_d0000_0"])

    # Try to insert an atoms object with a strange
    P = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    assert np.linalg.det(P) > 0
    atoms = make_supercell(bc_settings.prim_cell, P)

    atoms.symbols[0] = 'Cu'
    newstruct.insert_structure(init_struct=atoms)

    # Test that the generate with many templates works
    newstruct.generate_gs_structure_multiple_templates(num_prim_cells=16,
                                                       num_steps_per_temp=100,
                                                       eci={
                                                           'c0': 1.0,
                                                           'c3_d0000_0_000': -0.1
                                                       },
                                                       num_templates=2)

    # Try to export the ECI file
    fname = str(tmpdir / 'cf_func.csv')
    evaluator.export_dataset(fname)
    data = np.loadtxt(fname, delimiter=',')
    assert np.allclose(data[:, :-1], evaluator.cf_matrix)
    assert np.allclose(data[:, -1], evaluator.e_dft)
    os.remove(fname)
    os.remove(db_name)

    # Test load save
    fname = str(tmpdir / "demo_save.json")
    bc_settings.save(fname)
    bc_settings = settings_from_json(fname)
    os.remove(fname)


def test_initial_pool(db_name):
    basis_elements = [['Li', 'V'], ['X', 'O']]
    conc = Concentration(basis_elements=basis_elements)

    settings = CEBulk(concentration=conc,
                      crystalstructure="rocksalt",
                      a=4.0,
                      size=[2, 2, 1],
                      db_name=db_name,
                      max_cluster_size=3,
                      max_cluster_dia=[4.0, 4.0])
    ns = NewStructures(settings, struct_per_gen=2)
    ns.generate_initial_pool()

    # At this point there should be the following
    # structures in the DB
    expected_names = ["V1_O1_0", "Li1_X1_0", "V1_X1_0", "Li1_O1_0"]
    db = connect(db_name)
    for name in expected_names:
        num = sum(1 for row in db.select(name=name))
        assert num == 1


def test_1grouped_basis_probe(db_name):
    """Test a case where a grouped_basis is used with supercell."""
    # ------------------------------- #
    # 1 grouped basis                 #
    # ------------------------------- #
    # initial_pool + probe_structures #
    # ------------------------------- #
    basis_elements = [['Na', 'Cl'], ['Na', 'Cl']]
    conc = Concentration(basis_elements=basis_elements, grouped_basis=[[0, 1]])
    settings = CEBulk(concentration=conc,
                      crystalstructure="rocksalt",
                      a=4.0,
                      size=[2, 2, 1],
                      db_name=db_name,
                      max_cluster_size=3,
                      max_cluster_dia=[4.0, 4.0])

    assert settings.num_basis == 1
    assert len(settings.index_by_basis) == 1
    assert settings.spin_dict == {'Cl': 1.0, 'Na': -1.0}
    assert len(settings.basis_functions) == 1

    try:
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_random_structures()
        ns.generate_initial_pool()
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0,
                                    final_temp=0.001,
                                    num_temp=5,
                                    num_steps_per_temp=100,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))


def test_2grouped_basis_probe(db_name):
    # ------------------------------- #
    # 2 grouped basis                 #
    # ------------------------------- #
    # initial_pool + probe_structures #
    # ------------------------------- #
    basis_elements = [['Zr', 'Ce'], ['O'], ['O']]
    conc = Concentration(basis_elements=basis_elements, grouped_basis=[[0], [1, 2]])
    settings = CEBulk(concentration=conc,
                      crystalstructure="fluorite",
                      a=4.0,
                      size=[2, 2, 3],
                      db_name=db_name,
                      max_cluster_size=2,
                      max_cluster_dia=[4.01])
    settings.include_background_atoms = True
    fam_figures = get_figures_of_family(settings, "c2_d0005_0")
    assert len(fam_figures[0]) == 6
    assert len(fam_figures[1]) == 6
    assert len(fam_figures[2]) == 6
    assert settings.num_basis == 2
    assert len(settings.index_by_basis) == 2
    assert settings.spin_dict == {'Ce': 1.0, 'O': -1.0, 'Zr': 0}
    assert len(settings.basis_functions) == 2

    try:
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_initial_pool()
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0,
                                    final_temp=0.001,
                                    num_temp=5,
                                    num_steps_per_temp=100,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))

    # Try to create a cell with previously failing size
    size = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    atoms = make_supercell(settings.prim_cell, size)

    # This will fail if coordinatation number is wrong
    settings.set_active_template(atoms=atoms)


def test_2grouped_basis_bckgrnd_probe(db_name):
    # ---------------------------------- #
    # 2 grouped_basis + background atoms #
    # ---------------------------------- #
    # initial_pool + probe_structures    #
    # ---------------------------------- #
    basis_elements = [['Ca'], ['O', 'F'], ['O', 'F']]
    conc = Concentration(basis_elements=basis_elements, grouped_basis=[[0], [1, 2]])
    settings = CEBulk(concentration=conc,
                      crystalstructure="fluorite",
                      a=4.0,
                      size=[2, 2, 2],
                      db_name=db_name,
                      max_cluster_size=3,
                      max_cluster_dia=[4.01, 4.01])
    assert settings.num_basis == 2
    assert len(settings.index_by_basis) == 2
    assert settings.spin_dict == {'F': 1.0, 'O': -1.0}
    assert len(settings.basis_functions) == 1

    try:
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_initial_pool()
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0,
                                    final_temp=0.001,
                                    num_temp=5,
                                    num_steps_per_temp=100,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))


def test_fcc_binary_fixed_conc(db_name, mocker):
    # c_Au = 1/3 and c_Cu = 2/3
    A_eq = [[2, -1]]
    b_eq = [0]
    mocker.patch('clease.settings.ClusterExpansionSettings.create_cluster_list_and_trans_matrix')
    conc = Concentration(basis_elements=[['Au', 'Cu']], A_eq=A_eq, b_eq=b_eq)
    settings = CEBulk(concentration=conc,
                      crystalstructure='fcc',
                      a=3.8,
                      supercell_factor=27,
                      max_cluster_dia=5.0,
                      max_cluster_size=3,
                      db_name=db_name)

    # Loop through templates and check that all satisfy constraints
    atoms = settings.atoms
    num = len(atoms)
    ratio = num / 3.0
    assert ratio == pytest.approx(int(ratio))


def test_rocksalt_conc_fixed_one_basis(db_name, mocker):
    mocker.patch('clease.settings.ClusterExpansionSettings.create_cluster_list_and_trans_matrix')
    basis_elem = [['Li', 'X'], ['O', 'F']]
    A_eq = [[0, 0, 3, -2]]
    b_eq = [0]
    conc = Concentration(basis_elements=basis_elem, A_eq=A_eq, b_eq=b_eq)
    settings = CEBulk(concentration=conc,
                      crystalstructure='rocksalt',
                      a=3.8,
                      supercell_factor=27,
                      max_cluster_dia=5.0,
                      max_cluster_size=3,
                      db_name=db_name)

    atoms = settings.atoms
    num_O = sum(1 for atom in atoms if atom.symbol == 'O')
    ratio = num_O / 5.0
    assert ratio == pytest.approx(int(ratio))


def test_save_load_all_bf(db_name, tmpdir):
    conc = Concentration(basis_elements=[['Au', 'Cu', 'X']])
    settings = CEBulk(conc, db_name=db_name, max_cluster_size=2, max_cluster_dia=5.0, a=3.0)

    fname = str(tmpdir / 'save_load_all_bf.json')
    bfs = [
        Polynomial(settings.unique_elements),
        Trigonometric(settings.unique_elements),
        BinaryLinear(settings.unique_elements),
        BinaryLinear(settings.unique_elements, redundant_element='X')
    ]

    for bf in bfs:
        settings.basis_func_type = bf
        settings.save(fname)
        settings2 = settings_from_json(fname)
        bf2 = settings2.basis_func_type
        for k in bf.__dict__.keys():
            assert bf == bf2
    os.remove(fname)
