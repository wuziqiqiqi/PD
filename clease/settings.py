"""Definition of ClusterExpansionSetting Class.

This module defines the base-class for storing the settings for performing
Cluster Expansion in different conditions.
"""
import os
from itertools import product
from copy import deepcopy
import numpy as np
from ase.db import connect

from clease import _logger, LogVerbosity, ClusterExtractor
from clease.tools import (wrap_and_sort_by_position, indices2tags,
                          get_all_internal_distances, nested_list2str,
                          trans_matrix_index2tags)
from clease.basis_function import BasisFunction
from clease.template_atoms import TemplateAtoms
from clease.concentration import Concentration
from clease.trans_matrix_constructor import TransMatrixConstructor
from clease import AtomsManager
from clease.name_clusters import name_clusters
from clease.cluster_fingerprint import ClusterFingerprint
from clease.cluster import Cluster
from clease.cluster_list import ClusterList
from clease.template_filters import ValidConcentrationFilter


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
        self.num_basis = len(self.basis_elements)
        self.db_name = db_name
        self.size = to_3x3_matrix(size)

        self.prim_cell = self._get_prim_cell()
        self._tag_prim_cell()
        self._store_prim_cell()

        self.template_atoms = TemplateAtoms(supercell_factor=supercell_factor,
                                            size=self.size,
                                            skew_threshold=skew_threshold,
                                            db_name=self.db_name)
        self.atoms_mng = AtomsManager(None)

        self.max_cluster_size = max_cluster_size
        self.max_cluster_dia = self._format_max_cluster_dia(max_cluster_dia)
        self.all_elements = sorted([item for row in self.basis_elements for
                                    item in row])
        self.ignore_background_atoms = ignore_background_atoms
        self.background_indices = None
        self.num_elements = len(self.all_elements)
        self.unique_elements = sorted(list(set(deepcopy(self.all_elements))))
        self.num_unique_elements = len(self.unique_elements)
        self.index_by_basis = None

        self.index_by_trans_symm = []
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

    def unique_element_without_background(self):
        """Remove backgound elements."""
        if not self.ignore_background_atoms:
            bg_sym = set()
        else:
            bg_sym = set(x[0] for x in self.basis_elements if len(x) == 1)

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
        self.index_by_trans_symm = \
            self.atoms_mng.group_indices_by_trans_symmetry(self.prim_cell)
        if self.ignore_background_atoms:
            self.index_by_trans_symm = [x for x in self.index_by_trans_symm
                                        if x[0] not in self.background_indices]
        self.num_trans_symm = len(self.index_by_trans_symm)
        self.ref_index_trans_symm = [i[0] for i in self.index_by_trans_symm]

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
        """Add a tag to all the atoms in the unit cell to track the index."""
        for atom in self.prim_cell:
            atom.tag = atom.index

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

    def _get_max_cluster_dia(self, cell, ret_weights=False):
        lengths = []
        weights = []
        for w in product([-1, 0, 1], repeat=3):
            vec = cell.dot(w)
            if w == (0, 0, 0):
                continue
            lengths.append(np.sqrt(vec.dot(vec)))
            weights.append(w)

        # Introduce tolerance to make max distance strictly smaller than half
        # of the shortest cell dimension
        tol = 2 * 10**(-3)
        min_length = min(lengths) / 2.0
        min_length = min_length.round(decimals=3) - tol

        if ret_weights:
            min_indx = np.argmin(lengths)
            return min_length, weights[min_indx]
        return min_length

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

    def _check_max_cluster_dia(self, internal_distances):
        """
        Check that the maximum cluster diameter does not exactly correspond
        to an internal distance as this can lead to round off errors
        """
        for dia in self.max_cluster_dia:
            if np.min(np.abs(internal_distances - dia) < 1E-6):
                raise ValueError(
                    "One of the maximum cluster diameters correspond"
                    "to an internal distance. Try to increase the max"
                    "cluster diameter a tiny bit (for instance 4.0 -> 4.01"
                )

    def _get_supercell(self):
        # for atom in atoms:
        #     atom.tag = atom.index
        supercell = self.atoms_mng.close_to_cubic_supercell()
        max_cluster_dia_in_sc = \
            self._get_max_cluster_dia(supercell.get_cell().T)

        # Make large enough supercell (4 times max_cluster_dia can go inside)
        scale = int(4*np.max(self.max_cluster_dia)/max_cluster_dia_in_sc)
        if scale < 1:
            scale = 1
        supercell = supercell*(scale, scale, scale)
        supercell = wrap_and_sort_by_position(supercell)
        ref_indices = \
            self.atoms_mng.corresponding_indices(self.ref_index_trans_symm,
                                                 supercell)

        # Calculate the center of atomic positions in a supercell
        pos = supercell.get_positions()
        com = np.mean(pos, axis=0)

        # Calculate the center all the reference indices
        com_ref = np.mean(pos[ref_indices, :], axis=0)

        # Translate center of reference indices
        supercell.translate(com - com_ref)
        supercell.wrap()
        return supercell, ref_indices

    def _create_cluster_list(self):
        """Generate information for clusters."""
        self.cluster_list.clear()
        supercell, ref_indices = self._get_supercell()
        supercell.info['distances'] = \
            get_all_internal_distances(supercell, max(self.max_cluster_dia),
                                       ref_indices)
        self._check_max_cluster_dia(supercell.info['distances'])

        bkg_sc_indices = []
        if self.ignore_background_atoms:
            sc_manager = AtomsManager(supercell)
            bkg_sc_indices = \
                sc_manager.single_element_sites(self.basis_elements)

        extractor = ClusterExtractor(supercell)
        for size in range(2, self.max_cluster_size+1):
            all_clusters = []
            all_equiv_sites = []
            fingerprints = []
            for ref_indx in ref_indices:
                clusters = extractor.extract(ref_indx=ref_indx, size=size,
                                             cutoff=self.max_cluster_dia[size],
                                             ignored_indices=bkg_sc_indices)
                equiv_sites = \
                    [extractor.equivalent_sites(c[0]) for c in clusters]
                all_equiv_sites.append(equiv_sites)
                clusters = indices2tags(supercell, clusters)
                fingerprints += extractor.inner_prod
                all_clusters.append(clusters)
            names = name_clusters(fingerprints)
            self._update_cluster_list(names, all_clusters, all_equiv_sites,
                                      fingerprints, size)

        # Update with empty info
        for cluster in self.empty_clusters:
            self.cluster_list.append(cluster)

        # Update with singlet info
        for cluster in self.point_clusters:
            self.cluster_list.append(cluster)

    def _update_cluster_list(self, names, clusters, equiv_sites, fingerprints,
                             size):
        counter = 0
        for trans_symm in range(len(clusters)):
            for cluster, equiv in zip(clusters[trans_symm],
                                      equiv_sites[trans_symm]):
                clst = Cluster(names[counter], size,
                               2*np.sqrt(fingerprints[counter][0]),
                               fingerprints[counter],
                               self.ref_index_trans_symm[trans_symm],
                               cluster, equiv, trans_symm)
                self.cluster_list.append(clst)
                counter += 1

    @property
    def empty_clusters(self):
        empty = []
        num_trans_symm = len(self.ref_index_trans_symm)
        for symm in range(num_trans_symm):
            empty.append(
                Cluster('c0', 0, 0.0, ClusterFingerprint([0.0]),
                        self.ref_index_trans_symm[symm], [], [], symm)
            )
        return empty

    @property
    def point_clusters(self):
        point = []
        num_trans_symm = len(self.ref_index_trans_symm)
        for symm in range(num_trans_symm):
            point.append(
                Cluster('c1', 1, 0.0, ClusterFingerprint([1.0]),
                        self.ref_index_trans_symm[symm], [], [], symm)
            )
        return point

    @property
    def multiplicity_factor(self):
        """Return the multiplicity factor of each cluster."""
        num_sites_in_group = [len(x) for x in self.index_by_trans_symm]
        return self.cluster_list.multiplicity_factors(num_sites_in_group)

    @property
    def all_cf_names(self):
        num_bf = len(self.basis_functions)
        return self.cluster_list.get_all_cf_names(num_bf)

    @property
    def num_cf(self):
        """Return the number of correlation functions."""
        return len(self.all_cf_names)

    def _get_symm_groups(self):
        symm_groups = -np.ones(len(self.atoms), dtype=int)

        for group, indices in enumerate(self.index_by_trans_symm):
            symm_groups[indices] = group
        return symm_groups.tolist()

    def _cutoff_for_tm_construction(self):
        """ Get cutoff radius for translation matrix construction."""
        indices = self.cluster_list.unique_indices
        # start with some small positive number
        max_dist = 0.1

        for ref in indices:
            # MIC distance is a lower bound for the distance used in the
            # cluster
            mic_distances = self.atoms.get_distances(ref, indices, mic=True)
            dist = np.max(mic_distances)
            if dist > max_dist:
                max_dist = dist

        # 0.5 * max_dist is the radius, but give a bit of buffer (0.1)
        max_dist *= 0.51
        return max_dist

    def create_cluster_list_and_trans_matrix(self):
        self._create_cluster_list()

        symm_group = self._get_symm_groups()
        tm_cutoff = self._cutoff_for_tm_construction()

        # For smaller cell we currently have no method to decide how large
        # cutoff we need in order to ensure that all indices in unique_indices
        # are included. We therefore just probe the cutoff and increase it by
        # a 1 angstrom until we achieve the what we want
        all_included = False
        counter = 0
        max_attempts = 1000
        supercell, ref_indices = self._get_supercell()

        # We need to get the symmetry groups of the supercell. We utilise that
        # the supercell is tagged
        symm_group_sc = [-1 for _ in range(len(supercell))]
        for atom in supercell:
            symm_group_sc[atom.index] = symm_group[atom.tag]

        # Make as efficient as possible by evaluating only a subset of
        # the indices
        indices = [-1 for _ in range(len(self.atoms))]
        for atom in supercell:
            if indices[atom.tag] == -1:
                indices[atom.tag] = atom.index

        unique_index_symm = self.cluster_list.unique_indices_per_group
        unique_index_symm = [set(x) for x in unique_index_symm]
        all_unique_indices = set(self.cluster_list.unique_indices)
        while not all_included and counter < max_attempts:
            try:
                tmc = TransMatrixConstructor(supercell, tm_cutoff)
                tm_sc = tmc.construct(ref_indices, symm_group_sc,
                                      indices=indices)

                # Map supercell indices to normal indices
                tm = trans_matrix_index2tags(tm_sc, supercell, indices=indices)
                for i, row in enumerate(tm):
                    _ = [row[k] for k in unique_index_symm[symm_group[i]]]

                # For a simpler data structure in calculator store some
                # additional data
                for i, row in enumerate(tm):
                    gr = symm_group[i]
                    diff = all_unique_indices - unique_index_symm[gr]
                    for i in diff:
                        row[i] = -1  # Put -1 as this should never be accessed
                all_included = True
            except (KeyError, IndexError):
                tm_cutoff += 3.0
            counter += 1

        if counter >= max_attempts:
            raise RuntimeError("Could not find a cutoff such that all "
                               "unique_indices are included")

        self.trans_matrix = [{k: int(row[k]) for k in
                              self.cluster_list.unique_indices} for row in tm]

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
