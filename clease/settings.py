"""Definition of ClusterExpansionSetting Class.

This module defines the base-class for storing the settings for performing
Cluster Expansion in different conditions.
"""
import os
from itertools import product
from copy import deepcopy
import numpy as np
from ase.db import connect

from clease import _logger, LogVerbosity
from clease.tools import (wrap_and_sort_by_position, indices2tags,
                          get_all_internal_distances, nested_list2str,
                          trans_matrix_index2tags)
from clease.basis_function import BasisFunction
from clease.template_atoms import TemplateAtoms
from clease.concentration import Concentration
from clease import AtomsManager
from clease.name_clusters import name_clusters
from clease.cluster_fingerprint import ClusterFingerprint
from clease.cluster import Cluster
from clease.cluster_list import ClusterList
from clease.template_filters import ValidConcentrationFilter
from clease.cluster_manager import ClusterManager
from clease.tools import flatten


class ClusterExpansionSetting(object):
    """Base class for all Cluster Expansion settings."""

    def __init__(self, concentration, size=None, supercell_factor=27,
                 db_name='clease.db', max_cluster_size=4,
                 max_cluster_dia=[5.0, 5.0, 5.0], basis_function='polynomial',
                 skew_threshold=4, ignore_background_atoms=False):
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
        self.ignore_background_atoms = ignore_background_atoms
        self.num_basis = len(self.basis_elements)
        self.db_name = db_name
        self.size = to_3x3_matrix(size)

        self.prim_cell = self._get_prim_cell()
        self._tag_prim_cell()
        self._store_prim_cell()

        if self.ignore_background_atoms:
            self.cluster_manager = ClusterManager(self.prim_no_bkg())
        else:
            self.cluster_manager = ClusterManager(self.prim_cell)

        self.template_atoms = TemplateAtoms(supercell_factor=supercell_factor,
                                            size=self.size,
                                            skew_threshold=skew_threshold,
                                            db_name=self.db_name)
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

        # Set an active template (this ensures that index_by_basis etc is set)
        self._set_active_template_by_uid(0)

        # Run through the templates and filter out the ones that does not have
        # a valid compostion
        self.template_atoms.apply_filter(ValidConcentrationFilter(self))

        if self.template_atoms.num_templates == 0:
            raise RuntimeError('There are no templates the satisfies the '
                               'constraints')

        for uid in range(self.template_atoms.num_templates):
            self._set_active_template_by_uid(uid)
        # Set the initial template atoms to 0, which is the smallest cell
        self._set_active_template_by_uid(0)
        self._check_cluster_list_consistency()

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

        if not os.path.exists(db_name):
            self.create_cluster_list_and_trans_matrix()
            self._store_data()
        else:
            self._read_data()

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

    def prepare_new_active_template(self, uid):
        """Prepare necessary data structures when setting new template."""
        self.template_atoms_uid = uid
        atoms, self.size = \
            self.template_atoms.get_atoms(uid, return_size=True)
        self.atoms_mng.atoms = wrap_and_sort_by_position(atoms)

        self.index_by_basis = self._group_index_by_basis()

        self.background_indices = self._get_background_indices()
        self.index_by_sublattice = self.atoms_mng.index_by_tag()
        self.num_trans_symm = len(self.index_by_sublattice)
        self.ref_index_trans_symm = [i[0] for i in self.index_by_sublattice]

    def _set_active_template_by_uid(self, uid):
        """Set a fixed template atoms object as the active."""
        self.prepare_new_active_template(uid)

        # Read information from database
        # Note that if the data is not found, it will generate
        # the nessecary data structures and store them in the database
        self._read_data()

    def set_active_template(self, size=None, atoms=None,
                            generate_template=False):
        """Set a new template atoms object."""
        if size is not None and atoms is not None:
            raise ValueError("Specify either size or pass Atoms object.")
        if size is not None:
            uid = self.template_atoms.get_uid_with_given_size(
                size=size, generate_template=generate_template)
        elif atoms is not None:
            uid = self.template_atoms.get_uid_matching_atoms(
                atoms=atoms, generate_template=generate_template)
        else:
            uid = self.template_atoms.weighted_random_template()
        self._set_active_template_by_uid(uid)

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
        self.cluster_list = self.cluster_manager.info_for_template(self.atoms)

        if self.ignore_background_atoms:
            bkg_set = set(self.background_indices)
            for i in range(len(self.cluster_list)-1, -1, -1):
                index_set = set(flatten(self.cluster_list[i].indices))
                if len(bkg_set.intersection(index_set)) > 0:
                    del self.cluster_list[i]
            self.cluster_list.make_names_sequential()

        self.trans_matrix = self.cluster_manager.translation_matrix(self.atoms)

    def _store_data(self):
        size_str = nested_list2str(self.size)
        num = self.template_atoms_uid
        num_templates = self.template_atoms.num_templates
        _logger('Generating cluster data for template with size: {}. '
                '({} of {})'.format(size_str, num+1, num_templates),
                verbose=LogVerbosity.INFO)

        db = connect(self.db_name)
        data = {'cluster_list': [x.todict()
                                 for x in self.cluster_list.tolist()],
                'trans_matrix': self.trans_matrix}

        try:
            row = db.get(name="template", size=self._size2string())
            db.update(row.id, data=data)
        except KeyError:
            db.write(self.atoms, name='template', data=data,
                     size=self._size2string())

    def _keys2int(self, tm):
        """
        Convert the keys in the translation matrix to integers
        """
        tm_int = []
        for row in tm:
            tm_int.append({int(k): v for k, v in row.items()})
        return tm_int

    def _read_data(self):
        db = connect(self.db_name)
        try:
            select_cond = [('name', '=', 'template'),
                           ('size', '=', self._size2string())]
            row = db.get(select_cond)
            info_str = row.data.cluster_list
            self.cluster_list.clear()
            for item in info_str:
                cluster = Cluster.load(item)
                self.cluster_list.append(cluster)
            self.trans_matrix = self._keys2int(row.data.trans_matrix)
        except KeyError:
            self.create_cluster_list_and_trans_matrix()
            self._store_data()
        except (AssertionError, AttributeError, RuntimeError):
            self.reconfigure_settings()

    def _group_index_by_basis(self):
        first_symb_in_basis = [x[0] for x in self.basis_elements]
        return self.atoms_mng.index_by_symbol(first_symb_in_basis)

    def _activate_lagest_template(self):
        atoms = self.template_atoms.largest_template_by_diag
        self.set_active_template(atoms)

    def view_clusters(self):
        """Display all clusters along with their names."""
        from ase.gui.gui import GUI
        from ase.gui.images import Images
        self._activate_lagest_template()
        figures = self.cluster_list.get_figures(self.atoms)
        images = Images()
        images.initialize(figures)
        gui = GUI(images, expr='')
        gui.show_name = True
        gui.run()

    def reconfigure_settings(self):
        """Reconfigure templates stored in DB file."""
        # Reconfigure the cluster information for each template based on
        # current max_cluster_size and max_cluster_dia
        for uid in range(self.template_atoms.num_templates):
            self._set_active_template_by_uid(uid)
            self.create_cluster_list_and_trans_matrix()
            self._store_data()
        self._set_active_template_by_uid(0)
        _logger('Cluster data updated for all templates.\n'
                'You should also reconfigure DB entries (in CorrFunction '
                'class) to make the information on each structure to be '
                'consistent.',
                verbose=LogVerbosity.INFO)

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
        class_types = ['CEBulk', 'CECrystal']
        if type(self).__name__ not in class_types:
            raise NotImplementedError('Class {} '
                                      'is not supported.'
                                      ''.format(type(self).__name__))

        import json
        if type(self).__name__ == 'CEBulk':
            self.kwargs['classtype'] = 'CEBulk'
        else:
            self.kwargs['classtype'] = 'CECrystal'
        # Write keyword arguments necessary for initializing the class
        with open(filename, 'w') as outfile:
            json.dump(self.kwargs, outfile, indent=2)

    def _check_cluster_list_consistency(self):
        """Check that cluster names in all templates' info entries match."""
        db = connect(self.db_name)
        ref_clust_list = None
        for row in db.select(name='template'):
            cluster_list_str = row.data['cluster_list']
            cluster_list = ClusterList()
            for item in cluster_list_str:
                cluster = Cluster.load(item)
                cluster_list.append(cluster)
            cluster_list.sort()

            if ref_clust_list is None:
                ref_clust_list = cluster_list

            assert cluster_list == ref_clust_list


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
