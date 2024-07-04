"""Test to initiatialize CE using a CEBulk.

1. Initialize the CE
2. Add a few structures
3. Compute the energy
4. Run the evaluation routine
"""

import json
from pathlib import Path
import functools

import pytest
import numpy as np
from ase.calculators.emt import EMT
from ase.db import connect
from ase.build import bulk, make_supercell

from clease.settings import (
    CEBulk,
    settings_from_json,
    Concentration,
    ClusterExpansionSettings,
)
from clease import NewStructures, Evaluate
from clease.corr_func import CorrFunction
from clease.structgen import MaxAttemptReachedError
from clease.tools import update_db
from clease.basis_function import Polynomial, Trigonometric, BinaryLinear
from clease.cluster.cluster_list import ClusterDoesNotExistError

# If this is True, the JSON file containing the correlation functions
# Used to check consistency of the reference functions is updated
# This should normally be False
update_reference_file = False
tol = 1e-9
ref_file = Path(__file__).parent / "reference_corr_funcs_bulk.json"


def update_cf(new_cf):
    global ref_file
    with ref_file.open("w") as file:
        json.dump(new_cf, file, indent=2, separators=(",", ": "))


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


def calculate_cf(settings, atoms):
    cf = CorrFunction(settings)
    cf_dict = cf.get_cf(atoms)
    return cf_dict


@pytest.fixture
def make_settings(db_name, verify_clusters):
    def _make_settings(conc, verify=False, **kwargs):
        assert isinstance(conc, Concentration)
        defaults = dict(crystalstructure="fcc", a=4.05, db_name=db_name, max_cluster_dia=[5.0])
        defaults.update(**kwargs)
        settings = CEBulk(conc, **defaults)
        if verify:
            verify_clusters(settings)
        return settings

    return _make_settings


def test_load_from_db(make_conc, make_settings):
    basis_elements = [["Au", "Cu"]]
    conc = make_conc(basis_elements)
    settings = make_settings(conc)

    orig_atoms = settings.atoms.copy()
    atoms = bulk("Au", crystalstructure="fcc", a=4.05, cubic=True)
    settings.set_active_template(atoms=atoms)

    # Try to read back the old atoms
    settings.set_active_template(atoms=orig_atoms)


def test_max_cluster_dia(make_conc, make_settings):
    basis_elements = [["Au", "Cu"]]
    conc = make_conc(basis_elements)
    mcd = [4.3, 4.3, 4.3]
    mcd_orig = mcd.copy()
    settings = make_settings(conc, max_cluster_dia=mcd, verify=True)

    # Ensure max cluster dia has not been mutated
    assert mcd is not mcd_orig
    assert mcd == mcd_orig

    # Explicitly test format_max_cluster dia, ensure no mutation still
    out = settings.max_cluster_dia
    assert mcd is not out
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, mcd)


@pytest.mark.parametrize("mcs", [2, 3, 4, 5, 6, 7])
def test_max_cluster_size(mcs, make_conc, make_settings):
    """Test that we can construct clusters of arbitrary size"""
    basis = [["Au", "Cu"]]
    conc = make_conc(basis)
    # max_cluster_dia has length 1 shorter than max_cluster_size
    mcd = (mcs - 1) * [0.0]

    # We need to increase the diameter for 7-body slightly otherwise no clusters fit
    # within the cutoff radius.
    diameter = 5.0
    if mcs > 6:
        diameter += 0.5

    mcd[-1] = diameter  # We are only interested in clusters of a certain size
    settings = make_settings(conc, size=(3, 3, 3), max_cluster_dia=mcd, verify=True)
    assert settings.max_cluster_size == mcs

    max_found_size = max(cluster.size for cluster in settings.cluster_list.clusters)
    # Verify we actually found clusters of the max cluster size
    assert max_found_size == mcs

    atoms = settings.atoms.copy()
    atoms.symbols[[0, 3]] = "Cu"
    # This used to fail, when we only supported up to 4-body clusters
    calculate_cf(settings, atoms)


def test_corrfunc_au_cu(make_conc, make_settings, check_cf):
    basis_elements = [["Au", "Cu"]]
    conc = make_conc(basis_elements)
    conc = Concentration(basis_elements=basis_elements)
    settings = make_settings(conc, size=[3, 3, 3], max_cluster_dia=[4.3, 4.3, 4.3], verify=True)

    atoms = settings.atoms.copy()
    atoms.symbols[[0, 3]] = "Cu"

    cf = calculate_cf(settings, atoms)
    cf_key = "binary_fcc"

    check_cf(cf, cf_key)


def test_corrfunc_li_v_x_o(make_conc, make_settings, check_cf):
    basis_elements = [["Li", "V"], ["X", "O"]]
    conc = make_conc(basis_elements)
    settings = make_settings(
        conc,
        crystalstructure="rocksalt",
        a=4.0,
        size=[2, 2, 1],
        max_cluster_dia=[4.01, 4.01],
        verify=True,
    )
    atoms = settings.atoms.copy()
    Li_ind = [atom.index for atom in atoms if atom.symbol == "Li"]
    X_ind = [atom.index for atom in atoms if atom.symbol == "X"]
    atoms[Li_ind[0]].symbol = "V"
    atoms[X_ind[0]].symbol = "O"
    cf = calculate_cf(settings, atoms)
    cf_key = "two_basis"

    check_cf(cf, cf_key)


def test_corrfunc_nacl(make_conc, make_settings, check_cf):
    basis_elements = [["Na", "Cl"], ["Na", "Cl"]]
    conc = make_conc(basis_elements, grouped_basis=[[0, 1]])
    settings = make_settings(
        conc,
        crystalstructure="rocksalt",
        a=4.0,
        size=[2, 2, 1],
        max_cluster_dia=[4.01, 4.01],
    )
    atoms = settings.atoms.copy()
    atoms[1].symbol = "Cl"
    atoms[7].symbol = "Cl"
    cf = calculate_cf(settings, atoms)
    cf_key = "one_grouped_basis"

    check_cf(cf, cf_key)


def test_corrfunc_ca_o_f(make_conc, make_settings, check_cf):
    basis_elements = [["Ca"], ["O", "F"], ["O", "F"]]
    conc = make_conc(basis_elements, grouped_basis=[[0], [1, 2]])
    settings = make_settings(
        conc,
        crystalstructure="fluorite",
        a=4.0,
        size=[2, 2, 2],
        max_cluster_dia=[4.01, 4.01],
    )
    atoms = settings.atoms.copy()
    O_ind = [atom.index for atom in atoms if atom.symbol == "O"]
    atoms.symbols[O_ind[:2]] = "F"

    cf = calculate_cf(settings, atoms)
    cf_key = "two_grouped_basis_bckgrnd"

    check_cf(cf, cf_key)


def test_binary_system(make_conc, make_settings, make_tempfile):
    """Verifies that one can run a CE for the binary Au-Cu system.

    The EMT calculator is used for energy calculations
    """
    basis_elements = [["Au", "Cu"]]
    conc = make_conc(basis_elements)
    settings = make_settings(conc, size=[3, 3, 3], max_cluster_dia=[4.3, 4.3, 4.3])
    db_name = settings.db_name

    newstruct = NewStructures(settings, struct_per_gen=3)
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
    evaluator = Evaluate(settings, fitting_scheme="l2", alpha=1e-6)

    # Test subclusters for pairs
    for cluster in settings.cluster_list.get_by_size(2):
        sub_cl = settings.cluster_list.get_subclusters(cluster)
        sub_cl_name = set(c.name for c in sub_cl)
        assert sub_cl_name == set(["c0", "c1"])

    # Test a few known clusters. Triplet nearest neighbour
    tests = {
        "c3_d0000_0": set(["c0", "c1", "c2_d0000_0"]),
        "c3_d0001_0": set(["c0", "c1", "c2_d0000_0", "c2_d0001_0"]),
        "c4_d0000_0": set(["c0", "c1", "c2_d0000_0", "c3_d0000_0"]),
    }
    for name, expect in tests.items():
        triplet = settings.cluster_list.get_by_name(name)[0]
        sub_cl = settings.cluster_list.get_subclusters(triplet)
        sub_cl_name = set(c.name for c in sub_cl)
        assert sub_cl_name == expect

    # Try to insert an atoms object with a strange
    P = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    assert np.linalg.det(P) > 0
    atoms = make_supercell(settings.prim_cell, P)

    atoms.symbols[0] = "Cu"
    newstruct.insert_structure(init_struct=atoms)

    # Test that the generate with many templates works
    newstruct.generate_gs_structure_multiple_templates(
        num_prim_cells=16,
        num_steps_per_temp=100,
        eci={"c0": 1.0, "c3_d0000_0_000": -0.1},
        num_templates=2,
    )

    # Try to export the ECI file
    fname = make_tempfile("cf_func.csv")
    evaluator.export_dataset(fname)
    data = np.loadtxt(fname, delimiter=",")
    assert np.allclose(data[:, :-1], evaluator.cf_matrix)
    assert np.allclose(data[:, -1], evaluator.e_dft)

    # Test load save
    fname = make_tempfile("demo_save.json")
    settings.save(fname)
    settings_load = settings_from_json(fname)
    assert isinstance(settings_load, ClusterExpansionSettings)
    # We should add some more test assertions


def test_initial_pool(make_conc, make_settings):
    basis_elements = [["Li", "V"], ["X", "O"]]
    conc = make_conc(basis_elements)

    settings = make_settings(
        conc,
        crystalstructure="rocksalt",
        a=4.0,
        size=[2, 2, 1],
        max_cluster_dia=[4.1, 4.1],
    )
    ns = NewStructures(settings, struct_per_gen=2)
    ns.generate_initial_pool()

    # At this point there should be the following
    # structures in the DB
    expected_names = ["V1_O1_0", "Li1_X1_0", "V1_X1_0", "Li1_O1_0"]
    db = connect(settings.db_name)
    for name in expected_names:
        num = sum(1 for row in db.select(name=name))
        assert num == 1


def test_1grouped_basis_probe(make_conc, make_settings):
    """Test a case where a grouped_basis is used with supercell."""
    # ------------------------------- #
    # 1 grouped basis                 #
    # ------------------------------- #
    # initial_pool + probe_structures #
    # ------------------------------- #
    basis_elements = [["Na", "Cl"], ["Na", "Cl"]]
    conc = make_conc(basis_elements, grouped_basis=[[0, 1]])
    settings = make_settings(conc, crystalstructure="rocksalt", size=[2, 2, 1])

    assert settings.num_basis == 1
    assert len(settings.index_by_basis) == 1
    assert settings.spin_dict == {"Cl": 1.0, "Na": -1.0}
    assert len(settings.basis_functions) == 1

    try:
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_random_structures()
        ns.generate_initial_pool()
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_probe_structure(
            init_temp=1.0,
            final_temp=0.001,
            num_temp=5,
            num_steps_per_temp=100,
            approx_mean_var=True,
        )

    except MaxAttemptReachedError as exc:
        print(str(exc))


def test_2grouped_basis_probe(make_conc, make_settings):
    # ------------------------------- #
    # 2 grouped basis                 #
    # ------------------------------- #
    # initial_pool + probe_structures #
    # ------------------------------- #
    basis_elements = [["Zr", "Ce"], ["O"], ["O"]]
    conc = make_conc(basis_elements, grouped_basis=[[0], [1, 2]])
    settings = make_settings(
        conc,
        crystalstructure="fluorite",
        a=4.0,
        size=[2, 2, 3],
        max_cluster_dia=[4.01],
        include_background_atoms=True,
        verify=True,
    )
    fam_figures = get_figures_of_family(settings, "c2_d0005_0")
    assert len(fam_figures[0]) == 6
    assert len(fam_figures[1]) == 6
    assert len(fam_figures[2]) == 6
    assert settings.num_basis == 2
    assert len(settings.index_by_basis) == 2
    assert settings.spin_dict == {"Ce": 1.0, "O": -1.0, "Zr": 0}
    assert len(settings.basis_functions) == 2

    try:
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_initial_pool()
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_probe_structure(
            init_temp=1.0,
            final_temp=0.001,
            num_temp=5,
            num_steps_per_temp=100,
            approx_mean_var=True,
        )

    except MaxAttemptReachedError as exc:
        print(str(exc))

    # Try to create a cell with previously failing size
    size = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    atoms = make_supercell(settings.prim_cell, size)

    # This will fail if coordinatation number is wrong
    settings.set_active_template(atoms=atoms)


def test_2grouped_basis_bckgrnd_probe(make_conc, make_settings):
    # ---------------------------------- #
    # 2 grouped_basis + background atoms #
    # ---------------------------------- #
    # initial_pool + probe_structures    #
    # ---------------------------------- #
    basis_elements = [["Ca"], ["O", "F"], ["O", "F"]]
    conc = make_conc(basis_elements, grouped_basis=[[0], [1, 2]])
    settings = make_settings(
        conc,
        crystalstructure="fluorite",
        a=4.0,
        size=[2, 2, 2],
        max_cluster_dia=[4.01, 4.01],
        verify=True,
    )
    assert settings.num_basis == 2
    assert len(settings.index_by_basis) == 2
    assert settings.spin_dict == {"F": 1.0, "O": -1.0}
    assert len(settings.basis_functions) == 1

    try:
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_initial_pool()
        ns = NewStructures(settings, struct_per_gen=2)
        ns.generate_probe_structure(
            init_temp=1.0,
            final_temp=0.001,
            num_temp=5,
            num_steps_per_temp=100,
            approx_mean_var=True,
        )

    except MaxAttemptReachedError as exc:
        print(str(exc))


def test_fcc_binary_fixed_conc(make_conc, make_settings, mocker):
    mocker.patch("clease.settings.ClusterExpansionSettings.create_cluster_list_and_trans_matrix")

    # c_Au = 1/3 and c_Cu = 2/3
    A_eq = [[2, -1]]
    b_eq = [0]
    conc = make_conc([["Au", "Cu"]], A_eq=A_eq, b_eq=b_eq)
    settings = make_settings(conc, crystalstructure="fcc", supercell_factor=27)

    # Loop through templates and check that all satisfy constraints
    atoms = settings.atoms
    num = len(atoms)
    ratio = num / 3.0
    assert ratio == pytest.approx(int(ratio))

    assert settings.num_active_sublattices == 1
    assert settings.atomic_concentration_ratio == 1.0
    assert settings.ignored_species_and_conc == {}


def test_rocksalt_conc_fixed_one_basis(make_conc, make_settings, mocker):
    mocker.patch("clease.settings.ClusterExpansionSettings.create_cluster_list_and_trans_matrix")
    basis_elem = [["Li", "X"], ["O", "F"]]
    A_eq = [[0, 0, 3, -2]]
    b_eq = [0]
    conc = make_conc(basis_elem, A_eq=A_eq, b_eq=b_eq)
    settings = make_settings(conc, crystalstructure="rocksalt", supercell_factor=27)

    atoms = settings.atoms
    num_O = sum(atoms.symbols == "O")
    ratio = num_O / 5.0
    assert ratio == pytest.approx(int(ratio))

    assert settings.num_active_sublattices == 2
    assert settings.atomic_concentration_ratio == 1.0
    assert settings.ignored_species_and_conc == {}


def test_concentration_with_background(make_conc, make_settings):
    """Test recovering concentrations with ignored sublattices"""

    basis_elements = [["Ca"], ["O", "F"], ["O", "F"]]
    conc = make_conc(basis_elements, grouped_basis=[[0], [1, 2]])
    settings = make_settings(
        conc,
        crystalstructure="fluorite",
        a=4.0,
        size=[2, 2, 2],
        max_cluster_dia=[4.01, 4.01],
        verify=True,
    )

    assert settings.num_active_sublattices == 1
    assert settings.atomic_concentration_ratio == 2 / 3
    assert settings.ignored_species_and_conc == {"Ca": 1 / 3}

    basis_elements = [["Na", "Cl"], ["Na", "Cl"]]
    conc = make_conc(basis_elements, grouped_basis=[[0, 1]])
    settings = make_settings(
        conc,
        crystalstructure="rocksalt",
        a=4.0,
        size=[2, 2, 1],
        max_cluster_dia=[4.01, 4.01],
        verify=True,
    )

    assert settings.num_active_sublattices == 1
    assert settings.atomic_concentration_ratio == 1.0
    assert settings.ignored_species_and_conc == {}

    basis_elements = [["Au", "Cu"]]
    conc = make_conc(basis_elements)
    settings = make_settings(conc)

    assert settings.num_active_sublattices == 1
    assert settings.atomic_concentration_ratio == 1.0
    assert settings.ignored_species_and_conc == {}

    basis_elements = [["Zr", "Ce"], ["O"], ["O"]]
    conc = make_conc(basis_elements, grouped_basis=[[0], [1, 2]])
    settings = make_settings(
        conc,
        crystalstructure="fluorite",
        a=4.0,
        size=[2, 2, 3],
        max_cluster_dia=[4.01],
        verify=True,
    )

    assert settings.num_active_sublattices == 1
    assert settings.atomic_concentration_ratio == 1 / 3
    assert settings.ignored_species_and_conc == {"O": 2 / 3}

    # If the background atoms are included
    settings = make_settings(
        conc,
        crystalstructure="fluorite",
        a=4.0,
        size=[2, 2, 3],
        max_cluster_dia=[4.01],
        include_background_atoms=True,
        verify=True,
    )
    assert settings.num_active_sublattices == 2
    assert settings.atomic_concentration_ratio == 1.0
    assert settings.ignored_species_and_conc == {}

    # Two non-identical background sublattices
    basis_elements = [["Zr", "Ce"], ["O"], ["S"]]
    conc = Concentration(basis_elements)
    settings = make_settings(
        conc,
        crystalstructure="fluorite",
        a=4.0,
        size=[2, 2, 3],
        max_cluster_dia=[4.01],
        verify=True,
    )

    assert settings.num_active_sublattices == 1
    assert settings.atomic_concentration_ratio == 1 / 3
    assert settings.ignored_species_and_conc == {"O": 1 / 3, "S": 1 / 3}


@pytest.fixture
def au_cu_x_settings(make_conc, make_settings):
    """Simple settings with AuCuX for convenience"""
    conc = make_conc([["Au", "Cu", "X"]])
    return make_settings(conc)


@pytest.mark.parametrize(
    "bf_func",
    [
        Polynomial,
        Trigonometric,
        BinaryLinear,
        functools.partial(BinaryLinear, redundant_element="X"),
    ],
)
def test_save_load_all_bf_obj(bf_func, au_cu_x_settings, make_tempfile):
    settings = au_cu_x_settings  # Shorthand

    fname = make_tempfile("save_load_bf.json")

    bf = bf_func(settings.unique_elements)
    settings.basis_func_type = bf
    settings.save(fname)
    settings2 = settings_from_json(fname)
    bf2 = settings2.basis_func_type

    assert bf == bf2  # Test __eq__ operator
    assert vars(bf) == vars(bf2)  # Test "self" namespace


@pytest.mark.parametrize(
    "bf,expect",
    [
        # Test string input conversion
        ("polynomial", Polynomial),
        ("trigonometric", Trigonometric),
        ("binary_linear", BinaryLinear),
        ("POLYNOMIAL", Polynomial),  # We cast to .lower()
    ],
)
def test_basis_func_type_string(bf, expect, au_cu_x_settings):
    """Test setting basis func type with string inputs"""
    settings = au_cu_x_settings  # Shorthand
    # This must create the object
    settings.basis_func_type = bf
    assert isinstance(settings.basis_func_type, expect)
    assert settings.basis_func_type.unique_elements == settings.unique_elements


@pytest.mark.parametrize(
    "bf,expect",
    [
        # Test inputting existing bf functions
        (Polynomial(["Au", "Cu", "X"]), Polynomial),
        (Trigonometric(["Au", "Cu", "X"]), Trigonometric),
        (BinaryLinear(["Au", "Cu", "X"]), BinaryLinear),
        # Wrong order should be OK
        (Polynomial(["X", "Au", "Cu"]), Polynomial),
    ],
)
def test_basis_func_type_obj(bf, expect, au_cu_x_settings):
    """Test passing in objects as basis_func_type - should be the exact same
    object in memory."""
    settings = au_cu_x_settings  # Shorthand
    settings.basis_func_type = bf

    assert isinstance(settings.basis_func_type, expect)
    assert bf is settings.basis_func_type


@pytest.mark.parametrize(
    "bf",
    [
        "bad_input",  # Something random
        "binarylinear",  # misspelling
        Polynomial(["Mg", "Mn", "X"]),  # Wrong elements
        Polynomial(["Au", "Cu", "X", "Ag"]),  # Too many elements
        Polynomial(["Au", "Cu"]),  # Missing unique element
    ],
)
def test_basis_func_type_errors(bf, au_cu_x_settings):
    settings = au_cu_x_settings  # Shorthand
    with pytest.raises(ValueError):
        settings.basis_func_type = bf


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"max_cluster_dia": np.array([3, 4, 5])},
        {"include_background_atoms": True, "supercell_factor": 15},
    ],
)
def test_save_load_roundtrip(kwargs, make_conc, make_settings, make_tempfile, compare_dict):
    file = make_tempfile("settings.json")
    basis_elements = [["Au", "Cu"]]
    conc = make_conc(basis_elements)
    settings = make_settings(conc, **kwargs)

    settings.save(file)
    settings_loaded = ClusterExpansionSettings.load(file)

    compare_dict(settings.todict(), settings_loaded.todict())


@pytest.mark.parametrize("a_lp", np.linspace(3, 4, 8))
@pytest.mark.parametrize("crystalstructure", ["hcp", "rocksalt"])
def test_cutoff_equals_lp(crystalstructure, a_lp):
    """Having the cutoff be equal to the lattice parameter (or other internal distances)
    can cause a lot of issues. Here we test that we get a very specific and predictable
    error, since this numerical instability results in missing clusters that we would
    expect to find. See also !480."""
    if crystalstructure == "rocksalt":
        basis_elements = [["Na", "Au"], ["Cu", "Ag"]]
    else:
        basis_elements = [["Au", "Cu"]]
    cutoff_dia = [a_lp] * 3

    conc = Concentration(basis_elements=basis_elements)
    settings = CEBulk(
        conc,
        crystalstructure=crystalstructure,
        a=a_lp,
        max_cluster_dia=cutoff_dia,
        size=[2, 2, 2],
    )

    try:
        settings.ensure_clusters_exist()
    except ClusterDoesNotExistError as err:
        # This is the only error we tolerate is raised
        # Not all test settings will raise here either, but some probably will
        assert "try increasing the cutoff diameter" in str(err)
        print(f"Cluster failure with settings: {crystalstructure}, and {a_lp}.")


def test_assume_no_self_interactions(make_settings, make_conc):
    basis_elements = [["Au", "Cu"]]
    conc = make_conc(basis_elements)
    settings = make_settings(conc)
    atoms = settings.prim_cell.copy()
    settings.set_active_template(atoms)
    assert not settings.cluster_list.assume_no_self_interactions
    atoms = settings.prim_cell * (4, 4, 4)
    settings.set_active_template(atoms)
    assert settings.cluster_list.assume_no_self_interactions
