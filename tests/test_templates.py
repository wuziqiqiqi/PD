"""Test suite for TemplateAtoms."""
import pytest
import numpy as np
import ase
from ase.build import bulk
from ase.spacegroup import crystal
from ase.build import niggli_reduce
from ase.db import connect
from clease.settings import CEBulk, Concentration
from clease.settings import template_atoms
from clease.settings.template_atoms import TemplateAtoms
from clease.settings.template_filters import (
    AtomsFilter,
    CellFilter,
    SkewnessFilter,
    DistanceBetweenFacetsFilter,
    CellVectorDirectionFilter,
    ValidConcentrationFilter,
)
from clease.tools import wrap_and_sort_by_position


class SettingsPlaceHolder:
    """
    Dummy object that simply holds the few variables needed for the test.
    Only purpose of this is to make the test fast
    """

    atoms = None
    index_by_basis = []
    Concentration = None


class NumAtomsFilter(AtomsFilter):
    def __init__(self, min_num_atoms):
        self.min_num_atoms = min_num_atoms

    def __call__(self, atoms):
        return len(atoms) > self.min_num_atoms


class DummyCellFilter(CellFilter):
    def __call__(self, cell):
        return True


def get_settings_placeholder_valid_conc_filter(system):
    """
    Helper functions that initialises various dummy settings classes to be
    used together with the test_valid_conc_filter_class
    """
    settings = SettingsPlaceHolder()
    if system == "NaCl":
        prim_cell = bulk("NaCl", crystalstructure="rocksalt", a=4.0)
        settings.atoms = prim_cell
        settings.index_by_basis = [[0], [1]]

        # Force vacancy concentration to be exactly 2/3 of the Cl
        # concentration
        A_eq = [[0, 1, -2.0]]
        b_eq = [0.0]
        settings.concentration = Concentration(
            basis_elements=[["Na"], ["Cl", "X"]], A_eq=A_eq, b_eq=b_eq
        )

    elif system == "LiNiMnCoO":
        a = 2.825
        b = 2.825
        c = 13.840
        alpha = 90
        beta = 90
        gamma = 120
        spacegroup = 166
        basis_elements = [["Li"], ["Ni", "Mn", "Co"], ["O"]]
        basis = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 0.259)]

        A_eq = None
        b_eq = None

        conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq)
        prim_cell = crystal(
            symbols=["Li", "Ni", "O"],
            basis=basis,
            spacegroup=spacegroup,
            cellpar=[a, b, c, alpha, beta, gamma],
            size=[1, 1, 1],
            primitive_cell=True,
        )
        prim_cell = wrap_and_sort_by_position(prim_cell)
        settings.concentration = conc

        settings.index_by_basis = [[0], [2], [1, 3]]
        settings.atoms = prim_cell
    return settings


def check_NaCl_conc(templates):
    for atoms in templates:
        num_cl = sum(1 for atom in atoms if atom.symbol == "Cl")
        assert 2.0 * num_cl / 3.0 == pytest.approx(np.round(2.0 * num_cl / 3.0))
    return True


@pytest.fixture
def prim_cell():
    return bulk("Cu", a=4.05, crystalstructure="fcc")


@pytest.fixture
def template_atoms_factory(prim_cell):
    def _template_atoms_factory(**kwargs):
        default_settings = {"supercell_factor": 27, "size": None, "skew_threshold": 4}
        default_settings.update(**kwargs)
        return TemplateAtoms(prim_cell, **default_settings)

    return _template_atoms_factory


def test_fcc(template_atoms_factory):
    template_atoms = template_atoms_factory()
    templates = template_atoms.get_all_scaled_templates()
    ref = [
        [1, 1, 1],
        [1, 1, 2],
        [2, 2, 2],
        [2, 2, 3],
        [2, 2, 4],
        [2, 2, 5],
        [2, 3, 3],
        [2, 3, 4],
        [3, 3, 3],
    ]

    ref = [np.diag(x).tolist() for x in ref]
    sizes = [t.info["size"] for t in templates]
    assert ref == sizes


@pytest.mark.parametrize(
    "test",
    [
        {"system": "NaCl", "func": check_NaCl_conc},
        {"system": "LiNiMnCoO", "func": lambda templ: len(templ) >= 1},
    ],
)
def test_valid_concentration_filter(test):
    settings = get_settings_placeholder_valid_conc_filter(test["system"])

    template_generator = TemplateAtoms(
        settings.atoms, supercell_factor=20, skew_threshold=1000000000
    )

    conc_filter = ValidConcentrationFilter(settings.concentration, settings.index_by_basis)
    # Check that you cannot attach an AtomsFilter as a cell
    # filter
    with pytest.raises(TypeError):
        template_generator.add_cell_filter(conc_filter)

    template_generator.clear_filters()
    template_generator.add_atoms_filter(conc_filter)

    templates = template_generator.get_all_scaled_templates()
    assert test["func"](templates)


def test_dist_filter():
    f = DistanceBetweenFacetsFilter(4.0)
    cell = [[0.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cell = np.array(cell)
    assert not f(cell)
    cell[0, 0] = 0.3
    assert f(cell)


def test_fixed_vol(template_atoms_factory):
    template_atoms = template_atoms_factory()
    templates = template_atoms.get_fixed_volume_templates(num_prim_cells=4, num_templates=100)

    # Conform that the conventional cell is present
    found_conventional = False
    conventional = [4.05, 4.05, 4.05, 90, 90, 90]
    for atoms in templates:
        niggli_reduce(atoms)
        lengths_ang = atoms.cell.cellpar()
        if np.allclose(lengths_ang, conventional):
            found_conventional = True
            break
    assert found_conventional


def test_fixed_vol_with_conc_constraint(mocker, db_name):
    mocker.patch("clease.settings.ClusterExpansionSettings.create_cluster_list_and_trans_matrix")
    A_eq = [[3, -2]]
    b_eq = [0]
    conc = Concentration(basis_elements=[["Au", "Cu"]], A_eq=A_eq, b_eq=b_eq)

    settings = CEBulk(
        crystalstructure="fcc",
        a=3.8,
        size=[1, 1, 5],
        db_name=db_name,
        max_cluster_dia=[3.0],
        concentration=conc,
        supercell_factor=40,
    )
    settings.skew_threshold = 100

    tmp = settings.template_atoms

    sizes = [4, 5, 7, 10]
    valid_size = [5, 10]
    for s in sizes:
        templates = tmp.get_fixed_volume_templates(num_prim_cells=s)

        if s in valid_size:
            assert len(templates) > 0
        else:
            assert len(templates) == 0


def test_remove_atoms_filter(template_atoms_factory):
    template_atoms = template_atoms_factory(supercell_factor=3)

    f = NumAtomsFilter(16)
    template_atoms.add_atoms_filter(f)
    assert len(template_atoms.atoms_filters) == 1
    template_atoms.remove_filter(f)
    assert len(template_atoms.atoms_filters) == 0


def test_remove_cell_filter(template_atoms_factory):
    template_atoms = template_atoms_factory(supercell_factor=3)

    num_cell_filters = len(template_atoms.cell_filters)
    f = DummyCellFilter()
    template_atoms.add_cell_filter(f)
    assert len(template_atoms.cell_filters) == num_cell_filters + 1
    template_atoms.remove_filter(f)
    assert len(template_atoms.cell_filters) == num_cell_filters


def test_set_skewness_threshold(template_atoms_factory):
    template_atoms = template_atoms_factory()

    # Set the skewthreshold
    template_atoms.skew_threshold = 100

    # Check that the Skewness filter indeed has a value of 100
    for f in template_atoms.cell_filters:
        if isinstance(f, SkewnessFilter):
            assert f.ratio == 100


def test_size_and_supercell(template_atoms_factory):
    template_atoms = template_atoms_factory()
    assert template_atoms.size is None
    assert template_atoms.supercell_factor is not None

    template_atoms.size = [3, 3, 3]
    assert np.allclose(template_atoms.size, np.diag([3, 3, 3]))
    assert template_atoms.supercell_factor is None

    for _ in range(5):
        t = template_atoms.weighted_random_template()
        assert (t.cell == (template_atoms.prim_cell * (3, 3, 3)).get_cell()).all()
        assert t.info["size"] == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        assert t == t
        t_size = t

    template_atoms.supercell_factor = 27
    assert template_atoms.size is None
    assert template_atoms.supercell_factor == 27

    sizes = []
    for _ in range(5):
        t = template_atoms.weighted_random_template()
        assert t != t_size
        assert t == t
        size = t.info["size"]
        assert round(np.linalg.det(size)) <= template_atoms.supercell_factor
        sizes.append(size)

    for s0 in sizes:
        # At least 1 size should be different for each size.
        assert any(s0 != s for s in sizes)


def test_cell_direction_filter(db_name):
    cubic_cell = bulk("Cu", a=4.05, crystalstructure="fcc", cubic=True)
    db = connect(db_name)
    db.write(cubic_cell, name="primitive_cell")

    cell_filter = CellVectorDirectionFilter(cell_vector=2, direction=[0, 0, 1])

    template_atoms = TemplateAtoms(cubic_cell, supercell_factor=1, size=None, skew_threshold=40000)

    template_atoms.add_cell_filter(cell_filter)

    templates = template_atoms.get_fixed_volume_templates(num_prim_cells=5, num_templates=20)

    assert len(templates) > 1
    for temp in templates:
        _, _, a3 = temp.get_cell()
        assert np.allclose(a3[:2], [0.0, 0.0])


def test_iterate_one_template(template_atoms_factory):
    template_atoms = template_atoms_factory(supercell_factor=9)
    iterator = template_atoms.iterate_all_templates(max_per_size=1)
    # We should only ever have 1 size per template
    atoms_prev = next(iterator)
    count = 1
    for atoms_new in iterator:
        # Number of atoms should increase for each iteration
        assert len(atoms_new) > len(atoms_prev)
        count += 1
        atoms_prev = atoms_new
    # We won't necessarily get 1 template per size
    assert count > 1
    # This comes from checking the output.
    # if the algorithm changes in the future, this _may_ change
    # or if the settings in the test change
    assert count == 4


def test_iterate_all_templates(template_atoms_factory):
    template_atoms = template_atoms_factory(supercell_factor=6)

    count = 0
    for atoms_new in template_atoms.iterate_all_templates():
        assert isinstance(atoms_new, ase.Atoms)
        count += 1
    assert count > 1
    # Check the explicit method
    assert count == len(template_atoms.get_all_templates())
    # This comes from checking the output.
    # if the algorithm changes in the future, this _may_ change
    # or if the settings in the test change
    assert count == 3
