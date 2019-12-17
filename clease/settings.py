"""Definition of ClusterExpansionSetting Class.

This module defines the base-class for storing the settings for performing
Cluster Expansion in different conditions.
"""
from copy import deepcopy
import numpy as np
from ase.db import connect

from clease.tools import (wrap_and_sort_by_position,
                          nested_list2str)
from clease.basis_function import BasisFunction
from clease.template_atoms import TemplateAtoms
from clease.concentration import Concentration
from clease import AtomsManager
from clease.cluster_list import ClusterList
from clease.template_filters import ValidConcentrationFilter
from clease.cluster_manager import ClusterManager


class ClusterExpansionSetting(object):
    """Base class for all Cluster Expansion settings."""

    def __init__(self, prim, concentration, size=None, supercell_factor=27,
                 db_name='clease.db', max_cluster_size=4,
                 max_cluster_dia=[5.0, 5.0, 5.0], basis_function='polynomial',
                 skew_threshold=40, ignore_background_atoms=False):
        self.kwargs = {'size': size,
                       'supercell_factor': supercell_factor,
                       'db_name': db_name,
                       'max_cluster_size': max_cluster_size,
                       'max_cluster_dia': deepcopy(max_cluster_dia),
                       'ignore_background_atoms': ignore_background_atoms}

        if isinstance(concentration, Concentration):
            self.concentration = concentration
        elif isinstance(concentration, dict):
            self.concentration = Concentration.from_dict(concentration)
        else:
            raise TypeError("concentration has to be either dict or "
                            "instance of Concentration")

        self.kwargs["concentration"] = self.concentration.to_dict()
        self.cluster_list = ClusterList()
        self.basis_elements = deepcopy(self.concentration.basis_elements)
        self.num_basis = len(self.basis_elements)
        self._check_first_elements()
        self.ignore_background_atoms = ignore_background_atoms
        self.db_name = db_name
        self.size = to_3x3_matrix(size)

        self.prim_cell = prim
        self._tag_prim_cell()
        self._store_prim_cell()

        if self.ignore_background_atoms:
            self.cluster_manager = ClusterManager(self.prim_no_bkg())
        else:
            self.cluster_manager = ClusterManager(self.prim_cell)

        prim_mng = AtomsManager(prim)
        prim_ind_by_basis = prim_mng.index_by_symbol(
            [x[0] for x in self.basis_elements])
        conc_filter = ValidConcentrationFilter(concentration,
                                               prim_ind_by_basis)

        self.template_atoms = TemplateAtoms(
            self.prim_cell, supercell_factor=supercell_factor,
            size=self.size, skew_threshold=skew_threshold,
            filters=[conc_filter])
        self.atoms_mng = AtomsManager(None)

        self.max_cluster_size = max_cluster_size
        self.max_cluster_dia = self._format_max_cluster_dia(max_cluster_dia)
        self.cluster_manager.build(
            max_size=self.max_cluster_size,
            max_cluster_dia=self.max_cluster_dia
        )
        self.all_elements = sorted([item for row in self.basis_elements for
                                    item in row])
        self.background_indices = None
        self.num_elements = len(self.all_elements)
        self.unique_elements = sorted(list(set(deepcopy(self.all_elements))))
        self.num_unique_elements = len(self.unique_elements)
        self.index_by_basis = None

        self.index_by_sublattice = []
        self.ref_index_trans_symm = []
        self.template_atoms_uid = 0

        self.set_active_template(
            atoms=self.template_atoms.weighted_random_template())

        unique_element_no_bkg = self.unique_element_without_background()
        if isinstance(basis_function, BasisFunction):
            if basis_function.unique_elements != unique_element_no_bkg:
                raise ValueError("Unique elements in BasiFunction instance "
                                 "is different from the one in settings")
            self.bf_scheme = basis_function

        elif isinstance(basis_function, str):
            if basis_function.lower() == 'polynomial':
                from clease.basis_function import Polynomial
                self.bf_scheme = Polynomial(unique_element_no_bkg)
            elif basis_function.lower() == 'trigonometric':
                from clease.basis_function import Trigonometric
                self.bf_scheme = Trigonometric(unique_element_no_bkg)
            elif basis_function.lower() == "binary_linear":
                from clease.basis_function import BinaryLinear
                self.bf_scheme = BinaryLinear(unique_element_no_bkg)
            else:
                msg = "basis function scheme {} ".format(basis_function)
                msg += "is not supported."
                raise ValueError(msg)
        else:
            raise ValueError("basis_function has to be instance of "
                             "BasisFunction or a string")

        self.spin_dict = self.bf_scheme.spin_dict
        self.basis_functions = self.bf_scheme.basis_functions

        if len(self.basis_elements) != self.num_basis:
            raise ValueError("list of elements is needed for each basis")

    @property
    def atoms(self):
        return self.atoms_mng.atoms

    def prim_no_bkg(self):
        """
        Return an instance of the primitive cell where the background indices
        has been removed
        """
        prim = self.prim_cell.copy()
        bg_syms = self.get_bg_syms()
        delete = []
        for atom in prim:
            if atom.symbol in bg_syms:
                delete.append(atom.index)

        delete.sort(reverse=True)
        for i in delete:
            del prim[i]
        return prim

    def get_bg_syms(self):
        """
        Return the symbols in the basis where there is only one element
        """
        return set(x[0] for x in self.basis_elements if len(x) == 1)

    def unique_element_without_background(self):
        """Remove backgound elements."""
        if not self.ignore_background_atoms:
            bg_sym = set()
        else:
            bg_sym = self.get_bg_syms()

            # Remove bg_syms that are also present in basis with more than one
            # element
            for elems in self.basis_elements:
                if len(elems) == 1:
                    continue
                to_be_removed = set()
                for s in bg_sym:
                    if s in elems:
                        to_be_removed.add(s)

                bg_sym -= to_be_removed

        unique_elem = set()
        for x in self.basis_elements:
            unique_elem.update(x)
        return list(unique_elem - bg_sym)

    def _size2string(self):
        """Convert the current size into a string."""
        return nested_list2str(self.size)

    def prepare_new_active_template(self, template):
        """Prepare necessary data structures when setting new template."""
        self.template_atoms_uid = 0
        self.size = template.info['size']
        self.atoms_mng.atoms = wrap_and_sort_by_position(template)

        self.index_by_basis = self._group_index_by_basis()

        self.background_indices = self._get_background_indices()
        self.index_by_sublattice = self.atoms_mng.index_by_tag()
        self.num_trans_symm = len(self.index_by_sublattice)
        self.ref_index_trans_symm = [i[0] for i in self.index_by_sublattice]

    def set_active_template(self, size=None, atoms=None):
        """Set a new template atoms object."""
        if size is not None and atoms is not None:
            raise ValueError("Specify either size or pass Atoms object.")
        if size is not None:
            template = self.template_atoms.get_template_with_given_size(
                size=size)
        elif atoms is not None:
            template = self.template_atoms.get_template_matching_atoms(
                atoms=atoms)
        else:
            template = self.template_atoms.weighted_random_template()

        template = wrap_and_sort_by_position(template)

        if atoms is not None:
            # Check that the positions of the generated template
            # matches the ones in the passed object
            atoms = wrap_and_sort_by_position(atoms)
            if not np.allclose(template.get_positions(),
                               atoms.get_positions()):
                raise ValueError("Inconsistent positions. Passed object\n"
                                 "{}\nGenerated template\n{}"
                                 "".format(atoms.get_positions(),
                                           template.get_positions()))

        self.prepare_new_active_template(template)
        self.create_cluster_list_and_trans_matrix()

    def _tag_prim_cell(self):
        """
        Add a tag to all the atoms in the unit cell to track the sublattice.
        Tags are added such that that the lowest tags corresponds to "active"
        sites, while the highest tags corresponds to background sites. An
        example is a system having three sublattices that can be occupied by
        more than one species, and two sublattices that can be occupied by
        only one species. The tags 0, 1, 2 will then be assigned to the three
        sublattices that can be occupied by more than one species, and the two
        remaining lattices will get the tag 3, 4.
        """
        bg_sym = self.get_bg_syms()
        tag = 0

        # Tag non-background elements first
        for atom in self.prim_cell:
            if atom.symbol not in bg_sym:
                atom.tag = tag
                tag += 1

        # Tag background elements
        for atom in self.prim_cell:
            if atom.symbol in bg_sym:
                atom.tag = tag
                tag += 1

    def _store_prim_cell(self):
        """Store unit cell to the database."""
        db = connect(self.db_name)
        shape = self.prim_cell.get_cell_lengths_and_angles()
        for row in db.select(name='primitive_cell'):
            uc_shape = row.toatoms().get_cell_lengths_and_angles()
            if np.allclose(shape, uc_shape):
                return row.id

        uid = db.write(self.prim_cell, name='primitive_cell')
        return uid

    def _get_prim_cell(self):
        raise NotImplementedError("This function has to be implemented in "
                                  "in derived classes.")

    def _format_max_cluster_dia(self, max_cluster_dia):
        """Get max_cluster_dia in numpy array form"""
        # max_cluster_dia is list or array
        if isinstance(max_cluster_dia, (list, np.ndarray)):
            # Length should be either max_cluster_size+1 or max_cluster_size-1
            if len(max_cluster_dia) == self.max_cluster_size + 1:
                for i in range(2):
                    max_cluster_dia[i] = 0.
                max_cluster_dia = np.array(max_cluster_dia, dtype=float)
            elif len(max_cluster_dia) == self.max_cluster_size - 1:
                max_cluster_dia = np.array(max_cluster_dia, dtype=float)
                max_cluster_dia = np.insert(max_cluster_dia, 0, [0., 0.])
            else:
                raise ValueError("Invalid length for max_cluster_dia.")
        # max_cluster_dia is int or float
        elif isinstance(max_cluster_dia, (int, float)):
            max_cluster_dia *= np.ones(self.max_cluster_size - 1, dtype=float)
            max_cluster_dia = np.insert(max_cluster_dia, 0, [0., 0.])
        # Case for *None* or something else
        else:
            raise ValueError("max_cluster_dia must be float, int or list.")
        return max_cluster_dia.round(decimals=3)

    def _get_background_indices(self):
        """Get indices of the background atoms."""
        # check if any basis consists of only one element type
        basis = [i for i, b in enumerate(self.basis_elements) if len(b) == 1]

        bkg_indices = []
        for b_indx in basis:
            bkg_indices += self.index_by_basis[b_indx]
        return bkg_indices

    def _get_atoms(self):
        """Create atoms with a user-specified size."""
        atoms = self.prim_cell.copy() * self.size
        return wrap_and_sort_by_position(atoms)

    @property
    def multiplicity_factor(self):
        """Return the multiplicity factor of each cluster."""
        num_sites_in_group = [len(x) for x in self.index_by_sublattice]
        return self.cluster_list.multiplicity_factors(num_sites_in_group)

    @property
    def all_cf_names(self):
        num_bf = len(self.basis_functions)
        return self.cluster_list.get_all_cf_names(num_bf)

    @property
    def num_cf(self):
        """Return the number of correlation functions."""
        return len(self.all_cf_names)

    def create_cluster_list_and_trans_matrix(self):
        at_cpy = self.atoms
        self.cluster_list = self.cluster_manager.info_for_template(at_cpy)
        self.trans_matrix = self.cluster_manager.translation_matrix(at_cpy)

    def _keys2int(self, tm):
        """
        Convert the keys in the translation matrix to integers
        """
        tm_int = []
        for row in tm:
            tm_int.append({int(k): v for k, v in row.items()})
        return tm_int

    def _group_index_by_basis(self):
        first_symb_in_basis = [x[0] for x in self.basis_elements]
        return self.atoms_mng.index_by_symbol(first_symb_in_basis)

    def view_clusters(self):
        """Display all clusters along with their names."""
        from ase.gui.gui import GUI
        from ase.gui.images import Images
        figures = self.cluster_manager.get_figures()
        images = Images()
        images.initialize(figures)
        gui = GUI(images, expr='')
        gui.show_name = True
        gui.run()

    def get_all_templates(self):
        """
        Return a list with all template atoms
        """
        return self.template_atoms.get_all_templates()

    def view_templates(self):
        """
        Display all templates in the ASE GUi
        """
        from ase.visualize import view
        view(self.get_all_templates())

    def _check_first_elements(self):
        basis_elements = self.basis_elements
        num_basis = self.num_basis
        # This condition can be relaxed in the future
        first_elements = []
        for elements in basis_elements:
            first_elements.append(elements[0])
        if len(set(first_elements)) != num_basis:
            raise ValueError("First element of different basis should not be "
                             "the same.")

    def save(self, filename):
        """Write Setting object to a file in JSON format.

        Parameters:

        filename: str
            Name of the file to store the necessary settings to initialize
            the Cluster Expansion calculations.
        """

        import json
        with open(filename, 'w') as outfile:
            json.dump(self.kwargs, outfile, indent=2)


def to_3x3_matrix(size):
    if size is None:
        return None
    if isinstance(size, np.ndarray):
        size = size.tolist()

    # Is already a matrix
    if np.array(size).shape == (3, 3):
        return size

    if np.array(size).shape == (3,):
        return np.diag(size).tolist()

    raise ValueError("Cannot convert passed array {} to 3x3 matrix"
                     "".format(size))
