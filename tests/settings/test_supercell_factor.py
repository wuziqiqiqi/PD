import os
from clease.settings import CEBulk, CECrystal, Concentration


def test_fcc(db_name):
    conc = Concentration(basis_elements=[['Au', 'Cu']])

    settings = CEBulk(crystalstructure='fcc',
                      a=4.0,
                      supercell_factor=8,
                      concentration=conc,
                      db_name=db_name,
                      max_cluster_dia=3 * [4.01])
    settings.skew_threshold = 4
    settings.include_background_atoms = True

    scaled = settings.template_atoms.get_all_scaled_templates()
    assert len(scaled) == 3

    os.remove(db_name)

    settings = CEBulk(crystalstructure='fcc',
                      a=4.01,
                      supercell_factor=None,
                      size=[2, 2, 2],
                      concentration=conc,
                      db_name=db_name,
                      max_cluster_dia=3 * [4.0])
    settings.skew_threshold = 4
    settings.include_background_atoms = True

    scaled = settings.template_atoms.get_all_scaled_templates()
    assert len(scaled) == 1


def test_crystal(db_name):
    basis_elements = [['O', 'X'], ['O', 'X'], ['O', 'X'], ['Ta']]
    grouped_basis = [[0, 1, 2], [3]]
    concentration = Concentration(basis_elements=basis_elements, grouped_basis=grouped_basis)

    settings = CECrystal(basis=[(0., 0., 0.), (0.3894, 0.1405, 0.), (0.201, 0.3461, 0.5),
                                (0.2244, 0.3821, 0.)],
                         spacegroup=55,
                         cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                         supercell_factor=10,
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_dia=[3.0, 3.0])
    settings.basis_func_type = 'binary_linear'
    settings.skew_threshold = 4

    scaled = settings.template_atoms.get_all_scaled_templates()
    assert len(scaled) == 20
