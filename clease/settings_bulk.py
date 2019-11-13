"""Definitions of Cluster Expansion settings for bulk.

Cluster Expansion can be peformed on bulk material using either CEBulk
or CECrystal class.
"""
from ase.build import bulk
from ase.spacegroup import crystal
from clease.tools import wrap_and_sort_by_position
from clease.settings import ClusterExpansionSetting
from copy import deepcopy


class CEBulk(ClusterExpansionSetting):
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

    def __init__(self, concentration, crystalstructure='sc', a=None, c=None,
                 covera=None, u=None, size=None, supercell_factor=27,
                 db_name='clease.db', max_cluster_size=4,
                 max_cluster_dia=[5.0, 5.0, 5.0], basis_function='polynomial', skew_threshold=4, ignore_background_atoms=True):

        # Initialization
        self.structures = {'sc': 1, 'fcc': 1, 'bcc': 1, 'hcp': 1, 'diamond': 1,
                           'zincblende': 2, 'rocksalt': 2, 'cesiumchloride': 2,
                           'fluorite': 3, 'wurtzite': 2}
        self.crystalstructure = crystalstructure
        self.a = a
        self.c = c
        self.covera = covera
        self.u = u

        ClusterExpansionSetting.__init__(self, concentration, size,
                                         supercell_factor, db_name,
                                         max_cluster_size, max_cluster_dia,
                                         basis_function, skew_threshold,
                                         ignore_background_atoms)

        # Save raw input arguments for save/load. The arguments gets altered
        # during the initalization process to handle 'ignore_background_atoms'
        # case
        self.kwargs.update({'crystalstructure': crystalstructure,
                            'a': a,
                            'c': c,
                            'covera': covera,
                            'u': u})
        num_basis = len(self.concentration.orig_basis_elements)
        if num_basis != self.structures[self.crystalstructure]:
            msg = "{} has {} basis. ".format(
                self.crystalstructure, self.structures[self.crystalstructure])
            msg += "The number of basis specified by basis_elements is "
            msg += "{}".format(num_basis)
            raise ValueError(msg)

        self._check_first_elements()

    def _get_prim_cell(self):
        basis_elements = self.concentration.orig_basis_elements
        num_basis = len(basis_elements)
        if num_basis == 1:
            atoms = bulk(name='{}'.format(basis_elements[0][0]),
                         crystalstructure=self.crystalstructure, a=self.a,
                         c=self.c, covera=self.covera, u=self.u)

        elif num_basis == 2:
            atoms = bulk(name='{}{}'.format(basis_elements[0][0],
                                            basis_elements[1][0]),
                         crystalstructure=self.crystalstructure, a=self.a,
                         c=self.c, covera=self.covera, u=self.u)

        else:
            atoms = bulk(name='{}{}{}'.format(basis_elements[0][0],
                                              basis_elements[1][0],
                                              basis_elements[2][0]),
                         crystalstructure=self.crystalstructure, a=self.a,
                         c=self.c, covera=self.covera, u=self.u)
        atoms = wrap_and_sort_by_position(atoms)
        return atoms

    @staticmethod
    def load(filename):
        """Load settings from a file in JSON format.

        Parameters:

        filename: str
            Name of the file that has the settings.
        """
        import json
        with open(filename, 'r') as infile:
            kwargs = json.load(infile)
        classtype = kwargs.pop("classtype")
        if classtype != 'CEBulk':
            raise TypeError('Loaded setting file is not for CEBulk class')
        return CEBulk(**kwargs)


class CECrystal(ClusterExpansionSetting):
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

    def __init__(self, concentration, spacegroup=1, basis=None,
                 cell=None, cellpar=None, ab_normal=(0, 0, 1), size=None,
                 supercell_factor=27, db_name='clease.db', max_cluster_size=4,
                 max_cluster_dia=[5.0, 5.0, 5.0],
                 basis_function='polynomial', skew_threshold=4,
                 ignore_background_atoms=True):

        # Initialization
        self.spacegroup = spacegroup
        self.basis = basis
        self.cell = cell
        self.cellpar = cellpar
        self.ab_normal = ab_normal
        self.symbols = []
        num_basis = len(concentration.orig_basis_elements)
        for x in range(num_basis):
            self.symbols.append(concentration.orig_basis_elements[x][0])

        ClusterExpansionSetting.__init__(self, concentration, size,
                                         supercell_factor, db_name,
                                         max_cluster_size, max_cluster_dia,
                                         basis_function, skew_threshold,
                                         ignore_background_atoms)

        # Save raw input arguments for save/load. The arguments gets altered
        # during the initalization process to handle 'ignore_background_atoms'
        # case
        self.kwargs.update({'basis': deepcopy(basis),
                            'spacegroup': spacegroup,
                            'cell': cell,
                            'cellpar': cellpar,
                            'ab_normal': ab_normal})

        self._check_first_elements()

    def _get_prim_cell(self):
        atoms = crystal(symbols=self.symbols, basis=self.basis,
                        spacegroup=self.spacegroup, cell=self.cell,
                        cellpar=self.cellpar, ab_normal=self.ab_normal,
                        size=[1, 1, 1], primitive_cell=True)
        atoms = wrap_and_sort_by_position(atoms)
        return atoms

    @staticmethod
    def load(filename):
        """Load settings from a file in JSON format.

        Parameters:

        filename: str
            Name of the file that has the settings.
        """
        import json
        with open(filename, 'r') as infile:
            kwargs = json.load(infile)
        classtype = kwargs.pop("classtype")
        if classtype != 'CECrystal':
            raise TypeError('Loaded setting file is not for CEBulk class')
        return CEBulk(**kwargs)
