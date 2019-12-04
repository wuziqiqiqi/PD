"""Definitions of Cluster Expansion settings for bulk.

Cluster Expansion can be peformed on bulk material using either CEBulk
or CECrystal class.
"""
from ase.build import bulk
from ase.spacegroup import crystal
from clease.tools import wrap_and_sort_by_position
from clease.settings import ClusterExpansionSetting
from clease import Concentration
from copy import deepcopy
import json


def CEBulk(concentration, crystalstructure='sc', a=None, c=None,
           covera=None, u=None, size=None, supercell_factor=27,
           db_name='clease.db', max_cluster_size=4,
           max_cluster_dia=[5.0, 5.0, 5.0], basis_function='polynomial',
           skew_threshold=40, ignore_background_atoms=True):
    """
    Specify cluster expansion settings for bulk materials defined based on
    crystal structures.

    Parameters:

    concentration: Concentration object or dict
        Concentration object or dictionary specifying the basis elements and
        concentration range of constituting species

    crystalstructure: str
        Must be one of sc, fcc, bcc, hcp, diamond, zincblende, rocksalt,
        cesiumchloride, fluorite or wurtzite.

    a: float
        Lattice constant.

    c: float
        Lattice constant.

    covera: float
        c/a ratio used for hcp.  Default is ideal ratio: sqrt(8/3).

    u: float
        Internal coordinate for Wurtzite structure.

    size: list
        Size of the supercell (e.g., [2, 2, 2] for 2x2x2 cell).
        `supercell_factor` is ignored if both `size` and `supercell_factor`
        are specified.

    supercell_factor: int
        Maximum multipilicity factor for limiting the size of supercell
        created from the primitive cell. `supercell_factor` is ignored if
        both `size` and `supercell_factor` are specified.

    db_name: str
        Name of the database file

    max_cluster_size: int
        Maximum size (number of atoms in a cluster)

    max_cluster_dia: list of int or float
        A list of int or float containing the maximum diameter of clusters
        (in Å)

    basis_function: str
        One of "polynomial", "trigonometric" or "binary_linear"

    skew_threshold: int
        The maximum acceptable skew level (ratio of max and min diagonals of
        the cell) for supercells generated. A higher number allows highly
        skewed cells (e.g., 1x1x12) cells to be generated.

    ignore_background_atoms: bool
        if ``True``, a basis consisting of only one element type will be
        ignored when creating clusters.
    """
    structures = {'sc': 1, 'fcc': 1, 'bcc': 1, 'hcp': 1, 'diamond': 1,
                  'zincblende': 2, 'rocksalt': 2, 'cesiumchloride': 2,
                  'fluorite': 3, 'wurtzite': 2}

    num_basis = len(concentration.orig_basis_elements)
    if num_basis != structures[crystalstructure]:
        msg = "{} has {} basis. ".format(
            crystalstructure, structures[crystalstructure])
        msg += "The number of basis specified by basis_elements is "
        msg += "{}".format(num_basis)
        raise ValueError(msg)

    basis_elements = concentration.orig_basis_elements
    name = ''.join(x[0] for x in basis_elements)
    prim = bulk(name=name, crystalstructure=crystalstructure, a=a,
                c=c, covera=covera, u=u)
    prim = wrap_and_sort_by_position(prim)

    setting = ClusterExpansionSetting(
        prim, concentration, size, supercell_factor, db_name, max_cluster_size,
        max_cluster_dia, basis_function, skew_threshold,
        ignore_background_atoms)

    setting.kwargs.update(
        {'crystalstructure': crystalstructure, 'a': a,
         'c': c, 'covera': covera, 'u': u, 'factory': 'CEBulk'})
    return setting


def CECrystal(concentration, spacegroup=1, basis=None,
              cell=None, cellpar=None, ab_normal=(0, 0, 1), size=None,
              supercell_factor=27, db_name='clease.db', max_cluster_size=4,
              max_cluster_dia=[5.0, 5.0, 5.0],
              basis_function='polynomial', skew_threshold=40,
              ignore_background_atoms=True):
    """Store CE settings on bulk materials defined based on space group.

    Parameters:

    concentration: Concentration object or dict
        Concentration object or dictionary specifying the basis elements and
        concentration range of constituting species

    spacegroup: int | string | Spacegroup instance
        Space group given either as its number in International Tables
        or as its Hermann-Mauguin symbol.

    basis: list of scaled coordinates
        Positions of the unique sites corresponding to symbols given
        either as scaled positions or through an atoms instance.

    cell: 3x3 matrix
        Unit cell vectors.

    cellpar: [a, b, c, alpha, beta, gamma]
        Cell parameters with angles in degree. Is not used when `cell`
        is given.

    ab_normal: vector
        Is used to define the orientation of the unit cell relative
        to the Cartesian system when `cell` is not given. It is the
        normal vector of the plane spanned by a and b.

    size: list
        Size of the supercell (e.g., [2, 2, 2] for 2x2x2 cell).
        `supercell_factor` is ignored if both `size` and `supercell_factor`
        are specified.

    supercell_factor: int
        Maximum multipilicity factor for limiting the size of supercell
        created from the primitive cell. `supercell_factor` is ignored if
        both `size` and `supercell_factor` are specified.

    db_name: str
        name of the database file

    max_cluster_size: int
        maximum size (number of atoms in a cluster)

    max_cluster_dia: list of int or float
        A list of int or float containing the maximum diameter of clusters
        (in Å)

    basis_function: str
        One of "polynomial", "triogonometric" or "binary_linear"

    skew_threshold: int
        The maximum acceptable skew level (ratio of max and min diagonals of
        the cell) for supercells generated. A higher number allows highly
        skewed cells (e.g., 1x1x12) cells to be generated.

    ignore_background_atoms: bool
        if ``True``, a basis consisting of only one element type will be
        ignored when creating clusters.
    """

    symbols = []
    num_basis = len(concentration.orig_basis_elements)
    for x in range(num_basis):
        symbols.append(concentration.orig_basis_elements[x][0])

    prim = crystal(
        symbols=symbols, basis=basis, spacegroup=spacegroup, cell=cell,
        cellpar=cellpar, ab_normal=ab_normal, size=[1, 1, 1],
        primitive_cell=True)
    prim = wrap_and_sort_by_position(prim)

    setting = ClusterExpansionSetting(
        prim, concentration, size, supercell_factor, db_name,
        max_cluster_size, max_cluster_dia, basis_function, skew_threshold,
        ignore_background_atoms)
    setting.kwargs.update(
        {'basis': deepcopy(basis), 'spacegroup': spacegroup, 'cell': cell,
         'cellpar': cellpar, 'ab_normal': ab_normal,
         'factory': 'CECrystal'})
    return setting


def settingFromJSON(fname):
    """
    Initialise setting from JSON file.

    Parameters:

    fname: str
        JSON file where setting are stored
    """
    with open(fname, 'r') as infile:
        data = json.load(infile)

    factory = data.pop('factory')
    conc = Concentration.from_dict(data['concentration'])
    data['concentration'] = conc
    if factory == 'CEBulk':
        return CEBulk(**data)
    elif factory == 'CECrystal':
        return CECrystal(**data)
    raise ValueError('Unknown factory {}'.format(factory))
