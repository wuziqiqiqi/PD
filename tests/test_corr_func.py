"""Unit tests for the corr function class."""
import os
import pytest
from numpy.random import choice
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from clease.settings import CEBulk, Concentration
from clease.corr_func import CorrFunction, ClusterNotTrackedError
from clease import NewStructures
from clease.tools import wrap_and_sort_by_position


@pytest.fixture
def bc_settings(db_name):
    basis_elements = [["Au", "Cu", "Si"]]
    concentration = Concentration(basis_elements=basis_elements)
    return CEBulk(
        crystalstructure="fcc",
        a=4.05,
        size=[4, 4, 4],
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[5.73, 5.73],
    )


def get_mic_dists(atoms, cluster):
    """Get the MIC dist."""
    dists = []
    for indx in cluster:
        dist = atoms.get_distances(indx, cluster, mic=True)
        dists.append(dist)
    return dists


def test_trans_matrix(bc_settings):
    """Check that the MIC distance between atoms are correct."""
    db_name = bc_settings.db_name
    atoms = bc_settings.atoms
    tm = bc_settings.trans_matrix
    ref_dist = atoms.get_distance(0, 1, mic=True)
    for indx in range(len(atoms)):
        dist = atoms.get_distance(indx, tm[indx][1], mic=True)
        assert dist == pytest.approx(ref_dist)
    os.remove(db_name)


def test_supercell_consistency(db_name):
    basis_elements = [["Li", "X"], ["O", "X"]]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(
        crystalstructure="rocksalt",
        a=4.05,
        size=[1, 1, 1],
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[7.0, 4.0],
    )
    atoms = settings.atoms.copy()
    cf = CorrFunction(settings)
    cf_dict = cf.get_cf(atoms)

    atoms = wrap_and_sort_by_position(atoms * (4, 3, 2))
    cf_dict_sc = cf.get_cf(atoms)
    for k in cf_dict_sc.keys():
        assert cf_dict[k] == pytest.approx(cf_dict_sc[k])
    os.remove(db_name)


def test_error_message_for_non_existent_cluster(db_name):
    basis_elements = [["Li", "X"], ["O", "X"]]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(
        crystalstructure="rocksalt",
        a=4.05,
        size=[1, 1, 1],
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[7.0, 4.0],
    )
    corr = CorrFunction(settings)
    atoms = settings.atoms
    # No error should occure
    corr.get_cf_by_names(atoms, ["c3_d0000_0_000"])

    # Try a quadruplet: Have to raise error
    with pytest.raises(ClusterNotTrackedError):
        corr.get_cf_by_names(atoms, ["c4_d0001_0_0000"])
    os.remove(db_name)


def test_reconfigure(bc_settings):
    db_name = bc_settings.db_name
    newStruct = NewStructures(bc_settings)
    for i in range(10):
        atoms = bc_settings.atoms.copy()
        for a in atoms:
            a.symbol = choice(["Al", "Mg", "Si"])

        final = atoms.copy()
        calc = SinglePointCalculator(final, energy=-0.2)
        final.calc = calc
        newStruct.insert_structure(init_struct=atoms, final_struct=final)

    # Collect final_struct_ids
    db = connect(db_name)
    query = [("struct_type", "=", "initial")]
    final_str_ids = [row.final_struct_id for row in db.select(query)]

    cf = CorrFunction(bc_settings)
    cf.reconfigure_db_entries()

    # Confirm that the final_str_ids strays the same
    final_str_ids_rec = [row.final_struct_id for row in db.select(query)]
    assert final_str_ids == final_str_ids_rec
    os.remove(db_name)
