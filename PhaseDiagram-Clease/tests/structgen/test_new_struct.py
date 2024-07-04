import os
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from ase.io.trajectory import TrajectoryWriter
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.db import connect
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from clease.settings import CEBulk, ClusterExpansionSettings, Concentration
from clease import NewStructures


class CorrFuncPlaceholder:
    def get_cf(self, atoms):
        return {"c0": 0.0}


class BfSchemePlaceholder:
    name = "basis_func_type"


def test_insert_structures(db_name, tmpdir):
    settings_mock = MagicMock(spec=ClusterExpansionSettings)
    settings_mock.db_name = db_name
    connect(settings_mock.db_name).write(Atoms("H"))
    settings_mock.basis_func_type = BfSchemePlaceholder()

    # Mock several methods
    new_struct = NewStructures(settings_mock)
    new_struct._exists_in_db = MagicMock(return_value=False)
    new_struct._get_formula_unit = MagicMock(return_value="AuCu")
    new_struct.corrfunc = CorrFuncPlaceholder()
    new_struct._get_kvp = MagicMock(return_value={"name": "name"})

    symbols = ["Au", "Cu"]
    traj_in = str(tmpdir / "initial_structures.traj")
    traj_final = str(tmpdir / "final_structures.traj")
    traj_in_obj = TrajectoryWriter(traj_in)
    traj_final_obj = TrajectoryWriter(traj_final)

    num_struct = 10
    for i in range(num_struct):
        init = bulk("Au") * (5, 5, 5)
        symbols = np.random.choice(symbols, size=len(init))
        init.symbols = symbols

        final = init.copy()
        calc = EMT()
        final.calc = calc
        final.get_potential_energy()
        traj_in_obj.write(init)
        traj_final_obj.write(final)

    # Test when both initial and final is given
    new_struct.insert_structures(traj_init=traj_in, traj_final=traj_final)

    # Test when only initial is given
    new_struct.insert_structures(traj_init=traj_in)
    traj_in_obj.close()
    traj_final_obj.close()
    os.remove(traj_in)
    os.remove(traj_final)

    # Run some statistics
    assert new_struct._exists_in_db.call_count == 2 * num_struct
    assert new_struct._get_formula_unit.call_count == 2 * num_struct
    assert new_struct._get_kvp.call_count == 2 * num_struct

    # Check that final structures has a calculator
    db = connect(settings_mock.db_name)
    for row in db.select(struct_type="final"):
        assert row.calculator == "emt"
        energy = row.get("energy", None)
        assert energy is not None


def test_determine_generation_number(db_name):
    connect_f = lambda: connect(db_name)
    settings = MagicMock(spec=ClusterExpansionSettings, db_name=db_name, connect=connect_f)

    settings.db_name = db_name
    settings.connect().write(Atoms("H"))
    N = 5
    new_struct = NewStructures(settings, generation_number=None, struct_per_gen=N)

    def insert_in_db(n, gen):
        with settings.connect() as db:
            for _ in range(n):
                db.write(Atoms(), gen=gen)

    con = connect(db_name)
    # Verify we correctly wrote to the database
    assert con.count() == 1

    db_sequence = [
        {"num_insert": 0, "insert_gen": 0, "expect": 0},
        {"num_insert": 2, "insert_gen": 0, "expect": 0},
        {"num_insert": 3, "insert_gen": 0, "expect": 1},
        {"num_insert": 5, "insert_gen": 1, "expect": 2},
    ]
    tot = 1
    for i, action in enumerate(db_sequence):
        insert_in_db(action["num_insert"], action["insert_gen"])
        tot += action["num_insert"]
        assert con.count() == tot
        gen = new_struct._determine_gen_number()

        msg = "Test: #{} failed".format(i)
        msg += "Action: {}".format(action)
        msg += "returned generation: {}".format(gen)
        assert gen == action["expect"], msg


@patch("clease.structgen.new_struct.GSStructure")
def test_num_generated_structures(gs_mock, db_name):

    conc = Concentration(basis_elements=[["Au", "Cu"]])
    atoms = bulk("Au", a=2.9, crystalstructure="sc") * (5, 5, 5)
    atoms[0].symbol = "Cu"
    atoms[10].symbol = "Cu"

    def get_random_structure():
        atoms = bulk("Au", a=2.9, crystalstructure="sc") * (5, 5, 5)
        atoms.symbols = np.random.choice(["Au", "Cu"], size=len(atoms))
        atoms.calc = SinglePointCalculator(atoms, energy=0.0)
        return atoms, {"c1_0": 0.0}

    gs_mock.return_value.generate = get_random_structure
    gs_mock.return_value.min_energy = 0.0

    func = [
        {"func": NewStructures.generate_random_structures, "kwargs": {}},
        {
            "func": NewStructures.generate_gs_structure_multiple_templates,
            "kwargs": dict(
                num_templates=3,
                num_prim_cells=10,
                init_temp=2000,
                final_temp=1,
                num_temp=1,
                num_steps_per_temp=1,
                eci=None,
            ),
        },
        {"func": NewStructures.generate_initial_pool, "kwargs": {"atoms": atoms}},
        {
            "func": NewStructures.generate_gs_structure,
            "kwargs": dict(
                atoms=atoms,
                init_temp=2000,
                final_temp=1,
                num_temp=2,
                num_steps_per_temp=1,
                eci=None,
                random_composition=True,
            ),
        },
        {
            "func": NewStructures.generate_metropolis_trajectory,
            "kwargs": dict(atoms=atoms, random_comp=False),
        },
        {
            "func": NewStructures.generate_metropolis_trajectory,
            "kwargs": dict(atoms=atoms, random_comp=True),
        },
        {
            "func": NewStructures.generate_metropolis_trajectory,
            "kwargs": dict(atoms=None, random_comp=True),
        },
    ]

    tests = [
        {"gen": 0, "struct_per_gen": 5, "expect_num_to_gen": 5},
        {"gen": 0, "struct_per_gen": 8, "expect_num_to_gen": 3},
        {"gen": 1, "struct_per_gen": 2, "expect_num_to_gen": 2},
    ]

    # Patch the insert method such that we don't need to calculate the
    # correlation functions etc.
    def insert_struct_patch(cls, init_struct=None, final_struct=None, name=None, cf=None):
        atoms = bulk("Au")
        kvp = cls._get_kvp("Au")
        db = connect(db_name)
        if cf is None:
            db.write(atoms, kvp)
        else:
            db.write(atoms, kvp, external_tables={"tab_name": cf})

    def _get_formula_unit_patch(cls, *args, **kwargs):
        return "Au"

    with patch.object(NewStructures, "insert_structure", new=insert_struct_patch):
        with patch.object(NewStructures, "_get_formula_unit", new=_get_formula_unit_patch):
            for i, f in enumerate(func):
                settings = CEBulk(
                    conc,
                    max_cluster_dia=[3.0],
                    a=2.9,
                    crystalstructure="sc",
                    db_name=db_name,
                )
                for j, test in enumerate(tests):
                    msg = "Test #{} failed for func #{}".format(j, i)

                    new_struct = NewStructures(
                        settings,
                        generation_number=test["gen"],
                        struct_per_gen=test["struct_per_gen"],
                    )

                    num_to_gen = new_struct.num_to_gen()

                    special_msg = "Expect num in gen {}. Got: {}".format(
                        test["expect_num_to_gen"], num_to_gen
                    )

                    assert test["expect_num_to_gen"] == num_to_gen, msg + special_msg

                    # Call the current generation method
                    f["func"](new_struct, **f["kwargs"])

                    num_in_gen = new_struct.num_in_gen()
                    assert num_in_gen == test["struct_per_gen"], msg

                os.remove(db_name)  # Need to clear the database


def test_unique_name(db_name):
    connect_f = lambda: connect(db_name)
    settings = MagicMock(spec=ClusterExpansionSettings, db_name=db_name, connect=connect_f)
    settings.db_name = db_name
    settings.size = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    atoms = bulk("NaCl", crystalstructure="rocksalt", a=4.0)
    db = connect(db_name)
    # suffix 0, 3 and 9 missing
    suffixes = [1, 2, 4, 5, 6, 7, 8, 10, 11]
    with connect(db_name) as db:
        for suffix in suffixes:
            db.write(atoms, gen=0, name=f"Na1_Cl1_{suffix}", formula_unit="Na1_Cl1")

    N = 5
    new_struct = NewStructures(settings, generation_number=None, struct_per_gen=N)

    answer_list = ["Na1_Cl1_0", "Na1_Cl1_3", "Na1_Cl1_9", "Na1_Cl1_12", "Na1_Cl1_13"]

    for answer in answer_list:
        kvp = new_struct._get_kvp(formula_unit="Na1_Cl1")
        assert answer == kvp["name"]
        assert 1 == kvp["gen"]
        db.write(atoms, kvp)


@pytest.fixture
def new_struct_factory(db_name):
    def _new_struct_factory(
        basis_elements, crystalstructure, check_db=True, **kwargs
    ) -> NewStructures:
        conc = Concentration(basis_elements=basis_elements)
        settings = CEBulk(conc, crystalstructure=crystalstructure, db_name=db_name, **kwargs)
        return NewStructures(settings, check_db=check_db)

    return _new_struct_factory


@pytest.mark.parametrize("crystalstructure", ["sc", "fcc", "bcc"])
def test_make_conc_extrema_one_basis(crystalstructure, new_struct_factory):
    basis_elements = [["Au", "Ag"]]
    new_struct = new_struct_factory(basis_elements, crystalstructure, a=4)
    db_name = new_struct.settings.db_name
    con = connect(db_name)
    # Ensure id's don't exist already
    for i in (2, 3):
        with pytest.raises(KeyError):
            con.get(id=i)
    new_struct.generate_conc_extrema()
    # Select the extrema points
    for i in (2, 3):
        atoms = con.get(id=i).toatoms()
        sym_set = set(atoms.symbols)
        assert len(atoms) == 1
        assert len(sym_set) == 1
        assert sym_set.issubset({"Au", "Ag"})


def test_make_conc_extrema_two_basis(new_struct_factory):
    basis_elements = [["Au", "Ag"], ["Fe"]]
    crystalstructure = "rocksalt"
    new_struct = new_struct_factory(basis_elements, crystalstructure, a=4)

    db_name = new_struct.settings.db_name
    con = connect(db_name)
    # Ensure id's don't exist already
    for i in (2, 3):
        with pytest.raises(KeyError):
            con.get(id=i)
    new_struct.generate_conc_extrema()
    # Select the extrema points
    for i in (2, 3):
        atoms = con.get(id=i).toatoms()
        sym_set = set(atoms.symbols)
        # We should be able to make an extrema with just 2 elements
        assert len(atoms) == 2
        assert len(sym_set) == 2
        assert "Fe" in sym_set  # We only have 1 element in the second basis, always there
        assert sym_set.issubset({"Au", "Ag", "Fe"})


def test_check_db(new_struct_factory):
    """Test we can circumvent the DB check"""
    basis_elements = [["Au", "Ag"]]
    crystalstructure = "fcc"
    new_struct: NewStructures = new_struct_factory(
        basis_elements, crystalstructure, a=4, check_db=True
    )
    settings = new_struct.settings
    atoms = settings.prim_cell * (2, 2, 2)
    assert not new_struct._exists_in_db(atoms)
    new_struct.insert_structure(atoms)
    assert new_struct._exists_in_db(atoms)
    new_struct.check_db = False
    assert not new_struct._exists_in_db(atoms)


def test_insert_meta(new_struct_factory):
    basis_elements = [["Au", "Ag"]]
    crystalstructure = "fcc"
    new_struct: NewStructures = new_struct_factory(
        basis_elements,
        crystalstructure,
        a=4,
    )
    settings = new_struct.settings

    atoms = settings.prim_cell * (2, 2, 2)
    uid = new_struct.insert_structure(atoms, meta={"foo": "bar"})

    row = settings.connect().get(id=uid)
    assert row.foo == "bar"


def test_insert_initial_and_final(new_struct_factory, compare_atoms):
    basis_elements = [["Au", "Ag"]]
    crystalstructure = "fcc"
    new_struct: NewStructures = new_struct_factory(
        basis_elements,
        crystalstructure,
        a=4,
    )
    settings = new_struct.settings

    atoms = settings.prim_cell * (2, 2, 2)
    final = atoms.copy()
    calc = SinglePointCalculator(final, energy=123.321)
    final.calc = calc
    uid_ini, uid_final = new_struct.insert_structure(atoms, final_struct=final)

    con = settings.connect()
    ini_row = con.get(id=uid_ini)
    assert uid_final == ini_row.final_struct_id
    compare_atoms(ini_row.toatoms(), atoms)
    final_row = con.get(id=uid_final)
    assert final_row.energy == pytest.approx(final.get_potential_energy())
