"""Definition of ClusterExpansionSetting Class.

This module defines the base-class for storing the settings for performing
Cluster Expansion in different conditions.
"""
import os
from itertools import combinations, product
from copy import deepcopy
import numpy as np
from ase.db import connect

from clease import _logger, LogVerbosity, ClusterExtractor
from clease.floating_point_classification import FloatingPointClassifier
from clease.tools import (wrap_and_sort_by_position, flatten, dec_string,
                          indices2tags, get_all_internal_distances,
                          nested_list2str, trans_matrix_index2tags)
from clease.basis_function import BasisFunction
from clease.template_atoms import TemplateAtoms
from clease.concentration import Concentration
from clease.trans_matrix_constructor import TransMatrixConstructor
from clease import AtomsManager
from clease.name_clusters import name_clusters
from clease.cluster_fingerprint import ClusterFingerprint
from clease.cluster import Cluster
from clease.cluster_list import ClusterList


class ClusterExpansionSetting(object):
    """Base class for all Cluster Expansion settings."""

    def __init__(self, size=None, supercell_factor=None, concentration=None,
                 db_name=None, max_cluster_size=4, max_cluster_dia=None,
                 basis_function='sanchez', skew_threshold=4,
                 ignore_background_atoms=False):
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

        self.check_old_tm_algorithm = False
        self.kwargs["concentration"] = self.concentration.to_dict()
        self.cluster_list = ClusterList()
        self.basis_elements = deepcopy(self.concentration.basis_elements)
        self.num_basis = len(self.basis_elements)
        self.db_name = db_name
        self.size = to_3x3_matrix(size)
        self.cluster_info = []

        self.prim_cell = self._get_prim_cell()
        self._tag_prim_cell()
        self._store_prim_cell()
        self.float_max_dia, self.float_ang, self.float_dist = \
            self._init_floating_point_classifiers()


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

        self.cluster_info = []
        self.index_by_trans_symm = []
        self.ref_index_trans_symm = []
        self.kd_trees = None
        self.template_atoms_uid = 0
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
            if basis_function.lower() == 'sanchez':
                from clease.basis_function import Sanchez
                self.bf_scheme = Sanchez(unique_element_no_bkg)
            elif basis_function.lower() == 'vandewalle':
                from clease.basis_function import VandeWalle
                self.bf_scheme = VandeWalle(unique_element_no_bkg)
            elif basis_function.lower() == "sluiter":
                from clease.basis_function import Sluiter
                self.bf_scheme = Sluiter(unique_element_no_bkg)
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

        unique_elem = set()
        for x in self.basis_elements:
            unique_elem.update(x)
        return list(unique_elem - bg_sym)

    def _store_floating_point_classifiers(self):
        """Store classifiers in a separate DB entry."""
        from ase.atoms import Atoms
        db = connect(self.db_name)
        if sum(1 for row in db.select(name="float_classification")) >= 1:
            # Entry already exists
            return

        placeholder = Atoms()
        data = {"max_cluster_dia": self.float_max_dia.toJSON(),
                "angles": self.float_ang.toJSON(),
                "float_dist": self.float_dist.toJSON()}
        db.write(placeholder, data=data, name="float_classification")

    def _init_floating_point_classifiers(self):
        """Initialize the floating point classifiers from the DB if they
           exist, otherwise initialize a new one."""

        db = connect(self.db_name)
        try:
            row = db.get(name="float_classification")
            max_dia = row.data["max_cluster_dia"]
            angles = row.data["angles"]
            dists = row.data["float_dist"]
            float_max_dia = FloatingPointClassifier.fromJSON(max_dia)
            float_ang = FloatingPointClassifier.fromJSON(angles)
            dists = FloatingPointClassifier.fromJSON(dists)
        except KeyError:
            float_max_dia = FloatingPointClassifier(3)
            float_ang = FloatingPointClassifier(0)
            dists = FloatingPointClassifier(3)
        return float_max_dia, float_ang, dists

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
            bkg_sc_indices = sc_manager.single_element_sites(self.basis_elements)

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
            self._update_cluster_list(names, all_clusters, all_equiv_sites, fingerprints,
                                      size)

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

    # @property
    # def unique_indices(self):
    #     """Creates a list with the unique indices."""
    #     all_indices = deepcopy(self.ref_index_trans_symm)
    #     for item in self.cluster_info:
    #         for _, info in item.items():
    #             all_indices += flatten(info["indices"])
    #     return list(set(all_indices))

    # @property
    # def unique_indices_per_group(self):
    #     index_per_group = []
    #     for item in self.cluster_info:
    #         unique_indices = set()
    #         for _, info in item.items():
    #             unique_indices.update(flatten(info["indices"]))
    #         index_per_group.append(list(unique_indices))
    #     return index_per_group

    @property
    def multiplicity_factor(self):
        """Return the multiplicity factor of each cluster."""
        names = self.cluster_family_names
        mult_factor = {name: 0. for name in names}
        name_found = {name: False for name in names}
        normalization = {name: 0 for name in names}
        for name in names:
            for item in self.cluster_info:
                if name not in item.keys():
                    continue
                name_found[name] = True
                cluster = item[name]
                num_in_group = \
                    len(self.index_by_trans_symm[cluster["symm_group"]])
                mult_factor[name] += len(cluster["indices"]) * num_in_group
                normalization[name] += num_in_group

        for name in mult_factor.keys():
            mult_factor[name] = mult_factor[name] / normalization[name]
        for _, found in name_found.items():
            assert found
        return mult_factor

    def cluster_info_by_name(self, name):
        """Get info entries of all clusters with name."""
        name = str(name)
        info = []
        for item in self.cluster_info:
            if name in item.keys():
                info.append(item[name])
        return info

    def get_min_distance(self, cluster, positions):
        """Get minimum distances.

        Get the minimum distances between the atoms in a cluster according to
        dist_matrix and return the sorted distances (reverse order)
        """
        d = []
        for x in combinations(cluster, 2):
            x0 = positions[x[0], :]
            x1 = positions[x[1], :]
            d.append(self._get_distance(x0, x1))
        return np.array(sorted(d, reverse=True))

    def _get_distance(self, x0, x1):
        """Compute the Euclidean distance between two points."""
        diff = x1 - x0
        length = np.sqrt(diff.dot(diff))
        return length

    def indices_of_nearby_atom(self, ref_indx, size, pos):
        """Return the indices of the atoms nearby.

        Indices of the atoms are only included if distances smaller than
        specified by max_cluster_dia from the reference atom index.
        """
        nearby_indices = []
        dists = np.sqrt(np.sum((pos - pos[ref_indx, :])**2, axis=1))
        cutoff = self.max_cluster_dia[size]
        nearby_indices = np.nonzero(dists <= cutoff)[0].tolist()
        nearby_indices.remove(ref_indx)
        return nearby_indices

    @property
    def cluster_family_names(self):
        """Return a list of all cluster names."""
        families = []
        for item in self.cluster_info:
            families += list(item.keys())
        return list(set(families))

    @property
    def cluster_family_names_by_size(self):
        """Return a list of cluster familes sorted by size."""
        sort_list = []
        for item in self.cluster_info:
            for cname, c_info in item.items():
                # Sorted by:
                # 1) Number of atoms in cluster
                # 2) diameter (since cname starts with c3_04nn etc.)
                # 3) Unique descriptor
                sort_list.append((c_info["size"],
                                  cname.rpartition("_")[0],
                                  cname))
        sort_list.sort()
        sorted_names = []
        for item in sort_list:
            if item[2] not in sorted_names:
                sorted_names.append(item[2])
        return sorted_names

    @property
    def cluster_names(self):
        """Return the cluster names including decoration numbers."""
        names = ["c0"]
        bf_list = list(range(len(self.basis_functions)))
        for item in self.cluster_info:
            for name, info in item.items():
                if info["size"] == 0:
                    continue
                eq_sites = list(info["equiv_sites"])
                for dec in product(bf_list, repeat=info["size"]):
                    dec_str = dec_string(dec, eq_sites)
                    names.append(name + '_' + dec_str)
        return list(set(names))

    @property
    def num_clusters(self):
        """Return the number of clusters.

        Note: clusters with the same shape but with different decoration
              numbers are counted as a different cluster
        """
        return len(self.cluster_names)

    def cluster_info_given_size(self, size):
        """Get the cluster info of all clusters with a given size."""
        clusters = []
        for item in self.cluster_info:
            info_dict = {}
            for key, info in item.items():
                if info["size"] == size:
                    info_dict[key] = info
            clusters.append(info_dict)
        return clusters

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

    def create_cluster_info_and_trans_matrix(self):
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
        self.trans_matrix = [{k: row[k] for k in self.cluster_list.unique_indices}
                             for row in tm]

    def _store_data(self):
        size_str = nested_list2str(self.size)
        num = self.template_atoms_uid
        num_templates = self.template_atoms.num_templates
        _logger('Generating cluster data for template with size: {}. '
                '({} of {})'.format(size_str, num+1, num_templates),
                verbose=LogVerbosity.INFO)

        db = connect(self.db_name)
        data = {'cluster_list': self.cluster_list,
                'trans_matrix': self.trans_matrix}
        try:
            row = db.get(name="template", size=self._size2string())
            db.update(row.id, data=data)
        except KeyError:
            db.write(self.atoms, name='template', data=data,
                     size=self._size2string())

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
            # self.cluster_info = row.data.cluster_info
            # self._info_entries_to_list()
            self.trans_matrix = row.data.trans_matrix
        except KeyError:
            self.create_cluster_info_and_trans_matrix()
            self._store_data()
        except (AssertionError, AttributeError, RuntimeError):
            self.reconfigure_settings()

    def _get_name_indx(self, unique_name):
        size = int(unique_name[1])
        for symm in range(self.num_trans_symm):
            name_list = self.cluster_names[symm][size]
            try:
                n_indx = name_list.index(unique_name)
                return symm, n_indx
            except ValueError:
                continue

    def _group_index_by_basis(self):
        first_symb_in_basis = [x[0] for x in self.basis_elements]
        return self.atoms_mng.index_by_symbol(first_symb_in_basis)

    def view_clusters(self):
        """Display all clusters along with their names."""
        from ase.gui.gui import GUI
        from ase.gui.images import Images

        # set the active template which has the best chance of displaying
        # the largest diameter
        sizes = []
        for i in range(self.template_atoms.num_templates):
            atoms = self.template_atoms.get_atoms(i)
            lengths = atoms.get_cell_lengths_and_angles()[:3]
            sizes.append(lengths)

        uid = 0
        candidates = []
        for i, size in enumerate(sizes):
            if i == 0:
                candidates.append(min(size))
                uid = i
                continue
            if min(size) <= max(candidates):
                continue
            uid = i
        self._set_active_template_by_uid(uid)

        already_included_names = []
        cluster_atoms = []
        for unique_name in self.cluster_family_names_by_size:
            if unique_name in already_included_names:
                continue
            already_included_names.append(unique_name)
            for _, entry in enumerate(self.cluster_info):
                if unique_name in entry:
                    cluster = entry[unique_name]
                    break
            if cluster["size"] <= 1:
                continue
            name = cluster["name"]
            assert name == unique_name

            atoms = self.atoms.copy()

            keep_indx = cluster['indices'][0]
            equiv = list(cluster["equiv_sites"])

            for tag, indx in enumerate(keep_indx):
                atoms[indx].tag = tag
            if equiv:
                for group in equiv:
                    for i in range(1, len(group)):
                        atoms[keep_indx[group[i]]].tag = \
                            atoms[keep_indx[group[0]]].tag
            # atoms = create_cluster(atoms, keep_indx)

            # Extract the atoms in cluster for visualization
            atoms = atoms[keep_indx]
            positions = atoms.get_positions()
            cell = atoms.get_cell()
            origin = cell[0, :] + cell[1, :] + cell[2, :]
            supercell = atoms.copy()*(3, 3, 3)
            size = len(keep_indx)
            origin += positions[0, :]
            pos_sc = supercell.get_positions()
            lengths_sq = np.sum((pos_sc - origin)**2, axis=1)
            indices = np.argsort(lengths_sq)[:size]
            atoms = supercell[indices]

            atoms.info = {'name': name}
            cluster_atoms.append(atoms)

        images = Images()
        images.initialize(cluster_atoms)
        gui = GUI(images, expr='')
        gui.show_name = True
        gui.run()
        self._set_active_template_by_uid(0)

    def reconfigure_settings(self):
        """Reconfigure templates stored in DB file."""
        db = connect(self.db_name)
        # Reconfigure the float point classification based on setting
        ids = [row.id for row in db.select(name='float_classification')]
        db.delete(ids)
        self.float_max_dia, self.float_ang, self.float_dist = \
            self._init_floating_point_classifiers()

        # Reconfigure the cluster information for each template based on
        # current max_cluster_size and max_cluster_dia
        for uid in range(self.template_atoms.num_templates):
            self._set_active_template_by_uid(uid)
            self.create_cluster_info_and_trans_matrix()
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

    def _get_shortest_distance_in_unitcell(self):
        """Find the shortest distance between the atoms in the unit cell."""
        if len(self.prim_cell) == 1:
            lengths = self.prim_cell.get_cell_lengths_and_angles()[:3]
            return min(lengths)

        dists = []
        for ref_atom in range(len(self.prim_cell)):
            indices = list(range(len(self.prim_cell)))
            indices.remove(ref_atom)
            dists += list(self._cell.get_distances(ref_atom, indices,
                                                   mic=True))
        return min(dists)

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


    def subclusters(self, cluster_name):
        """Return all the sub-clusters of cluster."""
        info = self.cluster_info_by_name(cluster_name)

        sub_clst = []

        # Loop over symmetry groups
        for symm, item in enumerate(info):
            indices = item["indices"]
            size = item["size"]

            # Search in clusters up it the current size
            for s in range(size):
                clst_size = self.cluster_info_given_size(s)[symm]
                for _, v in clst_size.items():
                    if self._is_subcluster(v["indices"], indices):
                        sub_clst.append(v["name"])
        return list(set(sub_clst))

    def _is_subcluster(self, small_cluster, large_cluster):
        """Return True if small cluster is part of large cluster."""
        if len(small_cluster) == 0:
            # Singlet and empty is always a sub cluster
            return True

        if len(small_cluster[0]) >= len(large_cluster[0]):
            msg = "A cluster with size {} ".format(len(small_cluster[0]))
            msg += "cannot be a subcluster of a cluster with size "
            msg += "{}".format(len(large_cluster[0]))
            raise RuntimeError(msg)

        return any(set(s1).issubset(s2) for s1 in small_cluster
                   for s2 in large_cluster)


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
