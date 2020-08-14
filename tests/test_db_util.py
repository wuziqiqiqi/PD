import subprocess
import pytest
from ase.db import connect
from ase.build import bulk
from clease.db_util import get_all_cf_names, get_all_cf, get_cf_tables


def test_get_all_cf_names(db_name):
    cf = {'c0': 1.0, 'c1_1': 0.5, 'c2_d0000_0_00': 1.0}
    db = connect(db_name)
    for i in range(10):
        db.write(bulk('Cu'), external_tables={'polynomial_cf': cf})

    names = get_all_cf_names(db_name, 'polynomial_cf')
    assert sorted(names) == sorted(list(cf.keys()))


def test_get_cf(db_name):
    cf = {'c0': 1.0, 'c1_1': 0.5, 'c2_d0000_0_00': 1.0}
    db = connect(db_name)
    for i in range(10):
        db.write(bulk('Cu'), external_tables={'polynomial_cf': cf})

    cf_from_db = get_all_cf(db_name, 'polynomial_cf', 2)
    assert cf_from_db == cf


def test_get_cf_tables(db_name):
    cf = {'c0': 1.0, 'c1_1': 0.5, 'c2_d0000_0_00': 1.0}
    cf2 = {'c0': 0.1, 'c1_1': 0.5, 'c2_d0000_0_00': -0.2}
    db = connect(db_name)
    for _ in range(10):
        db.write(bulk('Cu'), external_tables={'polynomial_cf': cf, 'trigonometric_cf': cf2})

    cf_tab = get_cf_tables(db_name)
    expect_tab = ['polynomial_cf', 'trigonometric_cf']
    assert sorted(cf_tab) == expect_tab


def test_cli_runs(db_name):
    cf = {'c0': 1.0, 'c1_1': 0.5, 'c2_d0000_0_00': 1.0}
    cf2 = {'c0': 0.1, 'c1_1': 0.5, 'c2_d0000_0_00': -0.2}
    db = connect(db_name)
    for _ in range(10):
        db.write(bulk('Cu'), external_tables={'polynomial_cf': cf, 'trigonometric_cf': cf2})

    cmds = [["clease", "db", "-h"], ["clease", "db", db_name, "--show", "tab"],
            ["clease", "db", db_name, "--show", "names"],
            ["clease", "db", db_name, "--show", "cf", "--id", "1"]]
    for cmd in cmds:
        return_code = subprocess.call(cmd)
        assert return_code == 0
