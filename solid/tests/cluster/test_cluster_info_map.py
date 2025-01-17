import pytest
from ase.build import bulk
from ase.spacegroup import crystal

from clease.cluster import ClusterInfoMapper, AtomsNotContainedInLargeCellError
from clease.settings import CEBulk, CECrystal, Concentration
from clease.corr_func import CorrFunction
from clease.tools import wrap_and_sort_by_position


def dict_amost_equal(d1, d2):
    for k, v in d1.items():
        if v != pytest.approx(d2[k], abs=1e-6):
            return False
    return True


def test_fcc(db_name):
    basis_elements = [["Au", "Cu"]]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(
        crystalstructure="fcc",
        a=4.05,
        size=[6, 6, 6],
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[4.3, 4.3, 4.3],
    )

    # Let's try to obtain cluster info for conventional cell
    atoms_small = bulk("Au", crystalstructure="fcc", cubic=True, a=4.05)
    atoms_small = atoms_small * (2, 1, 1)
    atoms_small = wrap_and_sort_by_position(atoms_small)

    info_mapper = ClusterInfoMapper(settings.atoms, settings.trans_matrix, settings.cluster_list)

    map_info, map_tm = info_mapper.map_info(atoms_small)

    atoms_small[0].symbol = "Cu"
    atoms_small[4].symbol = "Cu"

    # Generate the cubic from scratch
    settings.set_active_template(atoms=atoms_small)

    cf = CorrFunction(settings)
    cf1 = cf.get_cf(atoms_small)

    # Change the info to the mapped one
    settings._cluster_list = map_info
    settings._trans_matrix = map_tm
    cf2 = cf.get_cf(atoms_small)

    assert dict_amost_equal(cf1, cf2)


def test_TaXO(db_name):
    basis_elements = [["O", "X"], ["Ta"], ["O", "X"], ["O", "X"]]
    grouped_basis = [[1], [0, 2, 3]]
    concentration = Concentration(basis_elements=basis_elements, grouped_basis=grouped_basis)

    bsg = CECrystal(
        basis=[
            (0.0, 0.0, 0.0),
            (0.2244, 0.3821, 0.0),
            (0.3894, 0.1405, 0.0),
            (0.201, 0.3461, 0.5),
        ],
        spacegroup=55,
        cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
        size=[4, 4, 4],
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[3.0, 3.0],
    )

    # Create a small cell
    atoms_small = crystal(
        basis=[
            (0.0, 0.0, 0.0),
            (0.2244, 0.3821, 0.0),
            (0.3894, 0.1405, 0.0),
            (0.201, 0.3461, 0.5),
        ],
        spacegroup=55,
        cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
        symbols=["O", "Ta", "O", "O"],
    )
    atoms_small = atoms_small * (2, 1, 1)

    atoms_small = atoms_small = wrap_and_sort_by_position(atoms_small)

    info_mapper = ClusterInfoMapper(bsg.atoms, bsg.trans_matrix, bsg.cluster_list)

    map_info, map_tm = info_mapper.map_info(atoms_small)

    # Swap 3 O with X
    count = 0
    for atom in atoms_small:
        if atom.symbol == "O":
            atom.symbol = "X"
            count += 1

        if count >= 3:
            break

    # Generate the cubic from scratch
    bsg.set_active_template(atoms=atoms_small)

    cf = CorrFunction(bsg)
    cf1 = cf.get_cf(atoms_small)

    # Change the info to the mapped one
    bsg._cluster_list = map_info
    bsg._trans_matrix = map_tm
    cf2 = cf.get_cf(atoms_small)

    assert dict_amost_equal(cf1, cf2)


def test_not_contained_error():
    atoms = bulk("Al", crystalstructure="fcc", a=4.05) * (2, 1, 1)
    atoms2 = bulk("Al", crystalstructure="fcc", a=4.05) * (1, 1, 3)

    info_mapper = ClusterInfoMapper(atoms, None, None)
    with pytest.raises(AtomsNotContainedInLargeCellError):
        info_mapper._map_indices(atoms2)
