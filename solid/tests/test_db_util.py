import subprocess
import pytest
from ase.db import connect
from ase.build import bulk
import clease
from clease import db_util, NewStructures
from clease.settings import CEBulk, Concentration
from clease.db_util import get_all_cf_names, get_all_cf, get_cf_tables


def test_get_all_cf_names(db_name):
    cf = {"c0": 1.0, "c1_1": 0.5, "c2_d0000_0_00": 1.0}
    db = connect(db_name)
    for i in range(10):
        db.write(bulk("Cu"), external_tables={"polynomial_cf": cf})

    names = get_all_cf_names(db_name, "polynomial_cf")
    assert sorted(names) == sorted(list(cf.keys()))


def test_get_cf(db_name):
    cf = {"c0": 1.0, "c1_1": 0.5, "c2_d0000_0_00": 1.0}
    db = connect(db_name)
    for i in range(10):
        db.write(bulk("Cu"), external_tables={"polynomial_cf": cf})

    cf_from_db = get_all_cf(db_name, "polynomial_cf", 2)
    assert cf_from_db == cf


def test_get_cf_tables(db_name):
    cf = {"c0": 1.0, "c1_1": 0.5, "c2_d0000_0_00": 1.0}
    cf2 = {"c0": 0.1, "c1_1": 0.5, "c2_d0000_0_00": -0.2}
    db = connect(db_name)
    for _ in range(10):
        db.write(bulk("Cu"), external_tables={"polynomial_cf": cf, "trigonometric_cf": cf2})

    cf_tab = get_cf_tables(db_name)
    expect_tab = ["polynomial_cf", "trigonometric_cf"]
    assert sorted(cf_tab) == expect_tab


def test_cli_runs(db_name):
    cf = {"c0": 1.0, "c1_1": 0.5, "c2_d0000_0_00": 1.0}
    cf2 = {"c0": 0.1, "c1_1": 0.5, "c2_d0000_0_00": -0.2}
    db = connect(db_name)
    for _ in range(10):
        db.write(bulk("Cu"), external_tables={"polynomial_cf": cf, "trigonometric_cf": cf2})

    cmds = [
        ["clease", "db", "--help"],
        ["clease", "db", "tab", db_name],
        ["clease", "db", "names", db_name],
        ["clease", "db", "cf", db_name, "1"],
    ]
    for cmd in cmds:
        return_code = subprocess.call(cmd)
        assert return_code == 0


@pytest.mark.parametrize(
    "meta",
    [
        {"mytable": {"test_A": 1, "test_B": 2}},
        {
            "table1": {"key1": "foo", "key2_": "bar"},
            "table2": {"key3": "fiz", "key4": "buz"},
        },
        # Test an empty meta input
        dict(),
    ],
)
def test_encode_decode_meta(meta):
    assert db_util.decode_meta(db_util.encode_meta(meta)) == meta


@pytest.mark.parametrize(
    "meta",
    [
        # Table name uses the delimiter
        {"mytable__": {"foo": "bar"}},
        # Table key uses the delimiter
        {"mytable": {"foo__": "bar"}},
    ],
)
def test_encode_bad_key(meta):
    with pytest.raises(db_util.InvalidMetaKey):
        db_util.encode_meta(meta)


@pytest.mark.parametrize(
    "table_name, key, expected",
    [
        ("poly_cf", "time", "poly_cf__time"),
        ("mytable", "whatever", "mytable__whatever"),
    ],
)
def test_encode_meta_key(table_name, key, expected):
    assert db_util.encode_meta_key(table_name, key) == expected


@pytest.mark.parametrize(
    "key",
    [
        "poly_cf",
        "time",
        db_util.MetaTableKeys.CLEASE_CONFIGURE_VERSION,
    ],
)
def test_decode_bad_key(key):
    with pytest.raises(db_util.InvalidMetaKey):
        db_util.decode_meta_key(key)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("poly_cf__time", ("poly_cf", "time")),
        ("mytable__anything", ("mytable", "anything")),
    ],
)
def test_decode_meta_key(key, expected):
    assert db_util.decode_meta_key(key) == expected


@pytest.fixture
def fast_settings(db_name):
    conc = Concentration([["Au", "Cu"]])
    defaults = dict(crystalstructure="fcc", a=4.05, db_name=db_name, max_cluster_dia=[3.0])
    settings = CEBulk(conc, **defaults)
    newstruct = NewStructures(settings, struct_per_gen=3)
    newstruct.generate_initial_pool()
    return settings


def test_check_version_ok(fast_settings):
    bf_name = fast_settings.basis_func_type.name
    tab_name = f"{bf_name}_cf"
    db_name = fast_settings.db_name

    with connect(db_name) as con:
        assert db_util.require_reconfigure_table(con, tab_name, 2, 3, 4) is False


@pytest.mark.parametrize("old_version", ["0.9", "0.10.0", "0.10"])
def test_check_outdated_version(old_version, fast_settings):
    bf_name = fast_settings.basis_func_type.name
    tab_name = f"{bf_name}_cf"
    db_name = fast_settings.db_name

    con = connect(db_name)
    # Artificially change the metadata table for a single row
    ext_tab = {
        db_util.MetaTableKeys.CLEASE_META_TABLE: {
            db_util.encode_meta_key(
                tab_name, db_util.MetaTableKeys.CLEASE_CONFIGURE_VERSION
            ): old_version
        }
    }

    con.update(2, external_tables=ext_tab)
    assert db_util.require_reconfigure_table(con, tab_name, 2, 3, 4)


def test_missing_meta_ext_table(fast_settings):
    bf_name = fast_settings.basis_func_type.name
    tab_name = f"{bf_name}_cf"
    db_name = fast_settings.db_name
    con = connect(db_name)

    con.delete_external_table(db_util.MetaTableKeys.CLEASE_META_TABLE)

    assert db_util.require_reconfigure_table(con, tab_name, 2, 3, 4)


def test_new_row_with_single_table(db_name):
    con = connect(db_name)
    tab_name = "poly_cf"

    atoms = bulk("Au", crystalstructure="fcc", a=4.0)
    cf = {"c0": 0.0}

    uid = db_util.new_row_with_single_table(con, atoms, tab_name, cf)

    row = con.get(id=uid)

    assert db_util.MetaTableKeys.CLEASE_META_TABLE in row
    assert tab_name in row


def test_new_row_with_many_tables(db_name):
    con = connect(db_name)
    tab1 = "poly_cf"
    tab2 = "some_other_cf"

    atoms = bulk("Au", crystalstructure="fcc", a=4.0)
    cf1 = {"c0": 0.0}
    cf2 = {"c0": 1.0}

    uid = db_util.new_row_with_many_tables(con, atoms, (tab1, tab2), (cf1, cf2))

    row = con.get(id=uid)

    assert db_util.MetaTableKeys.CLEASE_META_TABLE in row
    assert tab1 in row
    assert tab2 in row


def test_update_table(fast_settings):
    con = connect(fast_settings.db_name)
    row_id = 2
    table_name = "dummy_cf"
    cf = {"c0": 0.0, "c1": -1.5}

    db_util.update_table(con, row_id, table_name, cf)

    row = con.get(id=row_id)
    assert table_name in row
    meta = db_util.decode_meta(row[db_util.MetaTableKeys.CLEASE_META_TABLE])
    assert table_name in meta

    cf_loaded = row[table_name]
    assert cf == pytest.approx(cf_loaded)


def test_meta(fast_settings):
    con = connect(fast_settings.db_name)

    row = con.get(id=2)

    assert db_util.MetaTableKeys.CLEASE_META_TABLE in row
    meta = db_util.decode_meta(row[db_util.MetaTableKeys.CLEASE_META_TABLE])

    for tab_name, table in meta.items():
        assert tab_name.endswith("_cf")
        assert db_util.MetaTableKeys.TIME in table
        assert db_util.MetaTableKeys.CLEASE_CONFIGURE_VERSION in table
        assert table[db_util.MetaTableKeys.CLEASE_CONFIGURE_VERSION] == str(clease.__version__)
