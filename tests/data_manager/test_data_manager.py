import pytest
import numpy as np
from numpy.random import shuffle
from ase.db import connect
from ase.calculators.emt import EMT
from ase.build import bulk
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import clease
from clease import CorrFuncEnergyDataManager, CorrFuncVolumeDataManager
import clease.data_manager as dm
from clease.tools import update_db


@pytest.fixture
def simple_db(db_name):
    """Set up a simple database, fill it with some stuff and return the name.
    The fixture takes care of teardown after the test."""
    with connect(db_name) as db:
        atoms = bulk("Au")
        atoms2 = bulk("Au")

        cf_func = {"c0": 0.0, "c1_1": 1.0, "c2_d0000_0_00": -1.0}
        for i in range(10):
            uid = db.write(atoms, final_struct_id=2 * (i + 1), converged=1)
            clease.db_util.update_table(db, uid, "cf_func", cf_func)
            atoms2.calc = EMT()
            atoms2.get_potential_energy()
            db.write(atoms2)
    return db_name


@pytest.mark.parametrize(
    "manager_func,expect_header",
    [
        (
            CorrFuncEnergyDataManager,
            "# c0,c1_1,c2_d0000_0_00,E_DFT (eV/atom)",
        ),
        (
            CorrFuncVolumeDataManager,
            "# c0,c1_1,c2_d0000_0_00,Volume (A^3)",
        ),
    ],
)
def test_corr_final_energy(manager_func, expect_header, make_tempfile, simple_db):
    cf_names = ["c0", "c1_1", "c2_d0000_0_00"]

    manager = manager_func(simple_db, "cf_func", cf_names)

    X, y = manager.get_data([("converged", "=", 1)])

    expect_X = np.zeros((10, 3))
    expect_X[:, 0] = 0.0
    expect_X[:, 1] = 1.0
    expect_X[:, 2] = -1.0
    assert np.allclose(X, expect_X)
    assert manager.groups() == list(range(X.shape[0]))

    csvfile = make_tempfile("dataset.csv")
    manager.to_csv(csvfile)

    with open(csvfile, "r") as f:
        header = f.readline().strip()

    assert header == expect_header
    X_read = np.loadtxt(csvfile, delimiter=",")
    assert np.allclose(X, X_read[:, :-1])
    assert np.allclose(y, X_read[:, -1])

    # Add an initial structure that is by mistake labeled as converged
    with connect(simple_db) as db:
        cf = {k: 1.0 for k in cf_names}
        clease.db_util.new_row_with_single_table(db, Atoms(), "cf_func", cf, converged=True)

    with pytest.raises(dm.InconsistentDataError):
        X, y = manager.get_data([("converged", "=", 1)])


@pytest.mark.parametrize(
    "pattern,expect",
    [
        ("c", ["c0", "c1_1", "c2_d0000_0_00"]),
        ("c0", ["c0"]),
        ("d00", ["c2_d0000_0_00"]),
        ("0", ["c0", "c2_d0000_0_00"]),
    ],
)
def test_get_pattern(pattern, expect, simple_db):
    cf_names = ["c0", "c1_1", "c2_d0000_0_00"]
    manager = CorrFuncEnergyDataManager(simple_db, "cf_func", cf_names)
    manager.get_data([("converged", "=", 1)])

    res = manager.get_matching_names(pattern)
    assert res == expect


def test_get_cols(simple_db):
    cf_names = ["c0", "c1_1", "c2_d0000_0_00"]
    manager = CorrFuncEnergyDataManager(simple_db, "cf_func", cf_names)
    X, _ = manager.get_data([("converged", "=", 1)])

    tests = [
        {"names": ["c0", "c1_1", "c2_d0000_0_00"], "expect": X.copy()},
        {"names": ["c0"], "expect": X[:, 0]},
        {"names": ["c0", "c1_1"], "expect": X[:, :2]},
        {"names": ["c0", "c2_d0000_0_00"], "expect": X[:, [0, 2]]},
    ]

    for t in tests:
        res = manager.get_cols(t["names"])
        assert np.allclose(res, t["expect"])


@pytest.fixture
def shuffled_db(db_name):
    """
    Initialized a database where the entries are shuffled for the
    'test_consistent_order' test.
    """
    with connect(db_name) as db:
        init_struct_ids = []
        for i in range(10):
            atoms = bulk("Cu") * (3, 3, 3)
            cf_func = {"c0": np.random.rand(), "c1_1": np.random.rand()}
            dbId = db.write(atoms, converged=True, name=f"structure{i}")
            clease.db_util.update_table(db, dbId, "cf_func", cf_func)
            init_struct_ids.append(dbId)

    # We need to re-open the connection, to flush the db cache
    with connect(db_name) as db:
        # Add final structures in a random order
        shuffle(init_struct_ids)
        for init_id in init_struct_ids:
            atoms = db.get(id=init_id).toatoms()
            calc = SinglePointCalculator(atoms, energy=np.random.rand())
            atoms.calc = calc
            update_db(uid_initial=init_id, final_struct=atoms, db_name=db_name)

        cf_names = list(cf_func.keys())
    return db_name, cf_names


@pytest.mark.parametrize(
    "manager_func,target_col",
    [
        (CorrFuncEnergyDataManager, "energy"),
        (CorrFuncVolumeDataManager, "volume"),
    ],
)
@pytest.mark.parametrize("use_cf_names", [True, False])
def test_consistent_order(manager_func, target_col, use_cf_names, shuffled_db):
    db_name, cf_names = shuffled_db
    # Turn on/off telling the manager what the cf_names is
    manager_cf_names = cf_names if use_cf_names else None

    manager = manager_func(db_name, "cf_func", manager_cf_names)

    query = [("converged", "=", 1)]
    X, y = manager.get_data(query)

    # Extract via ASE calls
    X_ase = []
    y_ase = []
    with connect(db_name) as db:
        for row in db.select(query):
            x_row = [row["cf_func"][n] for n in cf_names]
            X_ase.append(x_row)

            fid = row.final_struct_id
            final_row = db.get(id=fid)
            y_ase.append(final_row[target_col] / final_row.natoms)

    assert np.allclose(X_ase, X)
    assert np.allclose(y, y_ase)


def test_final_volume_getter(db_name):
    with connect(db_name) as db:
        expected_volumes = []
        for i in range(10):
            init_struct = bulk("Cu")
            db.write(init_struct, final_struct_id=2 * i + 2)
            final_struct = init_struct.copy()
            db.write(final_struct)
            N = len(init_struct)
            expected_volumes.append(final_struct.get_volume() / N)

    final_vol_getter = dm.FinalVolumeGetter(db_name)
    ids = list(range(1, 21, 2))
    volumes = final_vol_getter.get_property(ids)
    assert np.allclose(expected_volumes, volumes)


def test_cf_vol_dep_eci(db_name):
    N = 10
    with connect(db_name) as db:
        volumes = []
        energies = []
        for i in range(N):
            init_struct = bulk("Cu", a=3.9 + 0.1 * i) * (1, 1, i + 1)
            cf = {"c0": 0.5, "c1_1": -1.0}

            clease.db_util.new_row_with_single_table(
                db, init_struct, "cf", cf, final_struct_id=2 * i + 2, converged=1
            )
            final_struct = init_struct.copy()
            calc = EMT()
            final_struct.calc = calc
            energy = final_struct.get_potential_energy()
            energies.append(energy / len(final_struct))
            volumes.append(final_struct.get_volume() / len(final_struct))
            db.write(final_struct)

    cf_getter = dm.CorrelationFunctionGetterVolDepECI(
        db_name, "cf", ["c0", "c1_1"], order=2, properties=["energy", "pressure"]
    )

    X, y = cf_getter.get_data([("converged", "=", 1)])

    expected_names = ["c0_V0", "c0_V1", "c0_V2", "c1_1_V0", "c1_1_V1", "c1_1_V2"]
    assert cf_getter._feat_names == expected_names

    X_expect = np.zeros((2 * N, 6))
    expect_groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert X_expect.shape == X.shape
    volumes = np.array(volumes)
    X_expect[:N, 0] = cf["c0"]
    X_expect[:N, 1] = cf["c0"] * volumes
    X_expect[:N, 2] = cf["c0"] * volumes**2
    X_expect[:N, 3] = cf["c1_1"]
    X_expect[:N, 4] = cf["c1_1"] * volumes
    X_expect[:N, 5] = cf["c1_1"] * volumes**2

    X_expect[N : 2 * N, 0] = 0.0
    X_expect[N : 2 * N, 1] = cf["c0"]
    X_expect[N : 2 * N, 2] = 2 * cf["c0"] * volumes
    X_expect[N : 2 * N, 3] = 0.0
    X_expect[N : 2 * N, 4] = cf["c1_1"]
    X_expect[N : 2 * N, 5] = 2 * cf["c1_1"] * volumes
    assert np.allclose(X, X_expect)
    assert cf_getter.groups() == expect_groups

    y_expect = np.zeros(2 * N)
    y_expect[:N] = energies
    assert np.allclose(y, y_expect)

    # Add bulk modulus to a few data points
    db.update(1, bulk_mod=100.5)
    db.update(5, bulk_mod=20.0)
    db.update(7, bulk_mod=56.5)

    # Extract data again
    cf_getter.properties = ("energy", "pressure", "bulk_mod")
    X, y = cf_getter.get_data([("converged", "=", 1)])

    y_bulk = [100.5, 20.0, 56.5]
    X_bulk = np.zeros((3, 6))
    X_bulk[:, 2] = 2 * cf["c0"] * volumes[[0, 2, 3]]
    X_bulk[:, 5] = 2 * cf["c1_1"] * volumes[[0, 2, 3]]
    X_expect = np.vstack((X_expect, X_bulk))
    y_expect = np.append(y_expect, y_bulk)
    expect_groups += [0, 2, 3]
    assert np.allclose(X, X_expect)
    assert np.allclose(y, y_expect)
    assert cf_getter.groups() == expect_groups

    # Extract with the pressure derivative
    db.update(1, dBdP=0.3)
    db.update(5, dBdP=4.0)
    db.update(7, dBdP=2.3)

    # Extract data again
    cf_getter.properties = ("energy", "pressure", "bulk_mod", "dBdP")
    X, y = cf_getter.get_data([("converged", "=", 1)])

    y_dBdP = np.array([0.3, 4.0, 2.3])
    X_dBdP = np.zeros((3, 6))
    X_dBdP[:, 2] = 2 * (1.0 + y_dBdP) * cf["c0"]
    X_dBdP[:, 5] = 2 * (1.0 + y_dBdP) * cf["c1_1"]
    X_expect = np.vstack((X_expect, X_dBdP))
    y_expect = np.append(y_expect, np.zeros(3))
    expect_groups += [0, 2, 3]
    assert np.allclose(X, X_expect)
    assert np.allclose(y, y_expect)
    assert cf_getter.groups() == expect_groups


@pytest.mark.parametrize(
    "test",
    [
        {
            "id_cf_names": {1: ["abc", "def", "ghi"], 2: ["abc", "def", "ghi"]},
            "matrix_repr": True,
            "min_common": set(["abc", "def", "ghi"]),
        },
        {
            "id_cf_names": {1: ["abc", "def", "ghi"], 2: ["abc", "def", "ghj"]},
            "matrix_repr": False,
            "min_common": set(["abc", "def"]),
        },
        {
            "id_cf_names": {1: ["abc", "def", "ghi"], 2: ["abc", "ghi"]},
            "matrix_repr": False,
            "min_common": set(["abc", "ghi"]),
        },
    ],
)
def test_is_matrix_representable(test):
    getter = dm.CorrelationFunctionGetter

    assert getter._is_matrix_representable(test["id_cf_names"]) == test["matrix_repr"]

    min_set = getter._minimum_common_cf_set(test["id_cf_names"])
    assert min_set == test["min_common"]


def test_cf_second_order(db_name):
    db = connect(db_name)
    cfs = [
        {
            "c1": 1.0,
            "c2": 1.0,
            "c3": 2.0,
        },
        {
            "c1": 1.0,
            "c2": 2.0,
            "c3": 4.0,
        },
    ]

    ids = []
    for cf in cfs:
        uid = clease.db_util.new_row_with_single_table(
            db, Atoms(), "polynomial_cf", cf, struct_type="initial"
        )
        ids.append(uid)

    getter = dm.CorrelationFunctionGetter(db_name, "polynomial_cf", order=2)
    X = getter.get_property(ids)
    expect = np.array([[1.0, 1.0, 2.0, 1.0, 2.0, 4.0], [1.0, 2.0, 4.0, 4.0, 8.0, 16.0]])
    assert np.allclose(X, expect)


def test_cf_reconfig_required(db_name):
    """Manually add external table, so we don't get a metadata.
    This should trigger an OutOfDateTable error.
    """
    db = connect(db_name)
    cfs = [
        {
            "c1": 1.0,
            "c2": 1.0,
            "c3": 2.0,
        },
        {
            "c1": 1.0,
            "c2": 2.0,
            "c3": 4.0,
        },
    ]

    ids = []
    for cf in cfs:
        uid = db.write(Atoms(), external_tables={"polynomial_cf": cf}, struct_type="initial")
        ids.append(uid)

    getter = dm.CorrelationFunctionGetter(db_name, "polynomial_cf", order=2)
    with pytest.raises(clease.db_util.OutOfDateTable):
        getter.get_property(ids)


@pytest.mark.parametrize(
    "prop, data",
    [
        ("my_custom_key", [100, 150, 175.21]),
        ("something_else", [200, 300, 400, 1, 2]),
    ],
)
def test_get_prop_getter(db_name, prop, data):
    db = connect(db_name)
    ids = []
    # Build the dataset - an initial which points to a final.
    # the "final" holds the actual property
    for val in data:
        kwargs = {prop: val}
        # Write the "final" with the desired property
        uid = db.write(Atoms(), **kwargs)
        # Write the "initial" which points to the final
        uid = db.write(Atoms(), final_struct_id=uid)
        ids.append(uid)  # Collect the "initial" ids

    getter = dm.FinalStructPropertyGetter(db_name, prop)
    X = getter.get_property(ids)
    assert pytest.approx(X) == data
