from typing import Sequence, Set, Dict, List, Iterator, Tuple, Callable
import logging
from itertools import product
import functools
from copy import deepcopy
import numpy as np
import ase

from clease import tools
from clease.datastructures import FourVector, Figure, TransMatrix

from .cluster_fingerprint import ClusterFingerprint
from .cluster import Cluster
from .cluster_list import ClusterList
from .cluster_generator import ClusterGenerator
from .utils import name_clusters, size

__all__ = ("ClusterManager",)

logger = logging.getLogger(__name__)


class ClusterManager:
    """
    Manager for construction of all cluster.

    Parameters:

    prim_cell: ase.Atoms
        Primitive cell
    """

    def __init__(self, prim_cell: ase.Atoms, background_syms: Set[str] = None):
        self._background_syms = background_syms or set()

        primitive_filtered = self._filter_background(prim_cell)

        self.generator = ClusterGenerator(primitive_filtered)
        self.clusters = ClusterList()
        self._cache = _CacheChecker()
        # Flag for determining if we can do the "trivial" path in trans_matrix
        # Should only be disabled for testing purposes!
        # Generally, should be set to True".
        self._allow_trivial_path = True

    @property
    def prim(self) -> ase.Atoms:
        """The primitive cell.
        Note, that background atoms have been removed from this atoms object."""
        return self.generator.prim

    @property
    def background_syms(self) -> Set[str]:
        """The symbols which are considered background."""
        return self._background_syms

    def _filter_background(self, atoms: ase.Atoms) -> ase.Atoms:
        """Filter the background atoms from an ASE Atoms object.

        Returns a copy of the original atoms object.
        """
        atoms = atoms.copy()

        # Find the indices we need to delete
        # If no background atoms are present, this will do nothing.
        mask = self.is_background_atoms(atoms)
        indices = np.arange(len(atoms))[mask]
        del atoms[indices]
        return atoms

    def is_background_atoms(self, atoms: ase.Atoms) -> np.ndarray:
        """Find every single background atom in an entire atoms object.

        Returns a NumPy array of booleans indicating if a site is a background site."""
        # Convert background syms to a list, as otherwise numpy treats the
        # set as an entire element.
        return np.isin(atoms.symbols, list(self.background_syms))

    def __eq__(self, other):
        return self.clusters == other.clusters and self.generator == other.generator

    def _prepare_new_build(self, max_cluster_dia):
        """Prepare for a new call to ``build``"""
        # Update the cache
        self._cache.set_cache(max_cluster_dia)
        # Clear any old clusters
        self.clusters.clear()

    def requires_build(self, max_cluster_dia: Sequence[float]) -> bool:
        """Check if the cluster manager requires a new build
        for a given set of cluster diameters"""
        return self._cache.requires_build(max_cluster_dia)

    def build(self, max_cluster_dia: Sequence[float]) -> None:
        """
        Construct all clusters.

        Parameters:

        max_cluster_dia: sequence of floats
            Maximum distance between two atoms in a cluster,
            for each cluster body. The diameters start at 2-body clusters
        """
        # Check if we already built the clusters with these settings
        if not self.requires_build(max_cluster_dia):
            return
        # We got a new set of settings, prepare to construct new clusters
        self._prepare_new_build(max_cluster_dia)

        # Number of lattices from 0 to N, where N
        # is the number of atoms in the primitive,
        # possibly without the background atoms.
        num_lattices = range(len(self.prim))
        all_fps: List[ClusterFingerprint] = []
        all_figures: List[List[Figure]] = []
        lattices = []

        for ref_lattice, (indx, diameter) in product(num_lattices, enumerate(max_cluster_dia)):
            cluster_size = indx + 2  # Size of cluster, start with 2-body at index 0
            figures, fps = self.generator.generate(cluster_size, diameter, ref_lattice)

            all_fps += fps
            all_figures += figures
            lattices += [ref_lattice] * len(figures)

        names = self._get_names(all_fps)
        # Transfer to the cluster list
        for figures, fp, name, ref_lattice in zip(all_figures, all_fps, names, lattices):
            # All figures are of the same size
            cluster_size = figures[0].size
            # Calculate the diameter from the first Figure, since they are all
            # geometrically equivalent, and thus have the same diameter.
            diameter = figures[0].get_diameter(self.prim)
            eq_sites = self.generator.equivalent_sites(figures[0])

            cluster = Cluster(
                name=name,
                size=cluster_size,
                diameter=diameter,
                fingerprint=fp,
                figures=figures,
                equiv_sites=eq_sites,
                group=ref_lattice,
            )
            self.clusters.append(cluster)

        # Add singlets
        for i in range(len(self.prim)):
            self.clusters.append(
                Cluster(
                    name="c1",
                    size=1,
                    diameter=0.0,
                    fingerprint=ClusterFingerprint([1.0]),
                    figures=[Figure([FourVector(0, 0, 0, i)])],
                    equiv_sites=[],
                    group=i,
                )
            )
        # Add empty
        self.clusters.append(
            Cluster(
                name="c0",
                size=0,
                diameter=0.0,
                fingerprint=ClusterFingerprint([0.0]),
                figures=[],
                equiv_sites=[],
                group=0,
            )
        )
        # Put the clusters in order of size. Has no practical effect,
        # but it looks nicer upon inspection.
        self.clusters.sort()

    @staticmethod
    def _get_names(all_fps: Sequence[ClusterFingerprint]):
        """
        Give a consistent name to all clusters

        Parameter:

        all_fps: list of ClusterFingerPrint
            A list with all the cluster fingerprints
        """
        # The following one-liner will do the job when we
        # decide to update cluster names from the legacy ones
        # where names where per by size
        # names = name_clusters(all_fps)

        sizes = sorted(set(size(fp) for fp in all_fps))
        names = [None for _ in all_fps]
        for s in sizes:
            fps = []
            indices = []
            for i, fp in enumerate(all_fps):
                if size(fp) == s:
                    indices.append(i)
                    fps.append(fp)
            names_per_size = name_clusters(fps)
            for i, n in zip(indices, names_per_size):
                names[i] = n
        return names

    def cluster_list_for_template(self, template: ase.Atoms) -> ClusterList:
        """
        Specialise the cluster information to a template. Returns a new
        ClusterList instance.

        Parameter:

        template: ase.Atoms
            Atoms object representing the simulation cell
        """
        unique = self.unique_four_vectors()
        lut = self.fourvec_to_indx(template, unique)
        ref_indices = [lut[FourVector(0, 0, 0, i)] for i in range(self.generator.num_sub_lattices)]

        template_cluster_list = deepcopy(self.clusters)
        for cluster in template_cluster_list:
            cluster.ref_indx = int(ref_indices[cluster.group])
            if cluster.size in {0, 1}:
                cluster.indices = []
            else:
                cluster.indices = self.generator.to_atom_index(cluster, lut)
        return template_cluster_list

    def unique_four_vectors(self) -> Set[FourVector]:
        """
        Return a list with all unique 4-vectors which are
        represented in any figure in all of the clusters.
        """
        # We utilize that FourVector objects are hashable,
        # and therefore can be filtered using a set()
        unique = set()
        for cluster in self.clusters:
            if cluster.size == 0:
                continue
            for figure in cluster.figures:
                for fv in figure.components:
                    unique.add(fv)
        return unique

    def get_figures(self) -> List[ase.Atoms]:
        """
        Return a list of atoms object representing the clusters
        """
        return self.clusters.get_figures(self.generator)

    def make_all_four_vectors(self, atoms: ase.Atoms) -> Tuple[np.ndarray, List[FourVector]]:
        """Construct the FourVector to every site which is not a background site.

        Returns an array of indices and a list with the corresponding FourVectors.
        Indices for background sites are skipped.
        """
        # Find every non-background site
        mask = ~self.is_background_atoms(atoms)
        # Array corresponding to the atomic index of sites we calculate the FVs for.
        indices = np.arange(len(atoms))[mask]
        tags = atoms.get_tags()[mask]
        pos = atoms.positions[mask]
        four_vectors = self.generator.many_to_four_vector(pos, tags)
        return indices, four_vectors

    def create_four_vector_lut(self, template: ase.Atoms) -> Dict[FourVector, int]:
        """
        Construct a lookup table (LUT) for the index in template given the
        wrapped vector

        Parameter:

        template: Atoms
            Atoms object to use when creating the lookup table (LUT)
        """
        indices, four_vectors = self.make_all_four_vectors(template)
        lut = {fv: idx for idx, fv in zip(indices, four_vectors)}
        return lut

    def fourvec_to_indx(
        self, template: ase.Atoms, unique: Sequence[FourVector]
    ) -> Dict[FourVector, int]:
        """Translate a set of unique FourVectors into their corresponding index
        in a template atoms object."""
        # Translate the four-vectors onto the corresponding index, by wrapping
        # them into the template atoms in cartesian space.
        cart_pos = self.generator.to_cartesian(*unique)

        cell = template.get_cell()
        cart_pos = tools.wrap_positions_3d(cart_pos, cell)

        unique_indices = []
        for i in range(cart_pos.shape[0]):
            diff_sq = np.sum((cart_pos[i, :] - template.positions) ** 2, axis=1)
            unique_indices.append(np.argmin(diff_sq))
        return dict(zip(unique, unique_indices))

    def translation_matrix(self, template: ase.Atoms) -> List[Dict[int, int]]:
        """
        Construct the translation matrix.

        The translation matrix translates a given atomic index to
        the corresponding atomic site if we started from index 0.

        Parameter:

        template: ase.Atoms
            Atoms object representing the simulation cell
        """
        # Get the unique four-vectors which are present in all of our clusters.
        unique = self.unique_four_vectors()

        wrap_fnc = self._prepare_wrap_fnc(template, unique)

        lut = self.create_four_vector_lut(template)

        # Map the un-translated unique 4-vectors to their index
        unique_indx_lut = self.fourvec_to_indx(template, unique)
        # call int() to convert from NumPy integer to python integer
        unique_index = [int(unique_indx_lut[u]) for u in unique]

        # The inverse of the LUT gives us the index of the template to its FourVector
        # Background atoms will be missing in this table.
        idx_to_fv = {idx: fv for fv, idx in lut.items()}

        def _make_site_mapping(index: int) -> Dict[int, int]:
            """Helper function to calculate the translation mapping for each
            atomic site."""
            try:
                vec = idx_to_fv[index]
            except KeyError:
                # Background site
                return {}

            # Calculate the four vectors wrapped back into the supercell
            four_vecs = wrap_fnc(vec)

            # Map the translated four-vectors to the unique index
            return {uidx: lut[fv] for uidx, fv in zip(unique_index, four_vecs)}

        # Calculate the mapping for each site in the template.
        trans_mat = [_make_site_mapping(i) for i in range(len(template))]
        return TransMatrix(trans_mat)

    def _prepare_wrap_fnc(
        self, template: ase.Atoms, unique: Sequence[FourVector]
    ) -> Callable[[FourVector], Iterator[FourVector]]:
        """Prepare the wrapping function which will map a given translation FourVector
        into the unique FourVectors with that site as a given reference.
        """
        # Only set "trivial_supercell" to True, if we allow that path.
        # Setting self._allow_trivial_path to False is only for testing purposes
        if self._allow_trivial_path:
            trivial_supercell = tools.is_trivial_supercell(self.prim, template)
        else:
            trivial_supercell = False

        # Set up the FourVector wrap function
        if trivial_supercell:
            rep_arr = tools.get_repetition(self.prim, template)
            # Pre-calculate the XYZ arrays of the unique FourVectors,
            # as well as the sublattices. We utilize the NumPy vectorization
            # for calculating the shift + modulo operation.
            unique_xyz = np.array([u.xyz_array for u in unique], dtype=int)
            sublattices = [u.sublattice for u in unique]
            wrap_fnc = functools.partial(
                self._wrap_four_vectors_trivial,
                unique_xyz=unique_xyz,
                sublattices=sublattices,
                rep_arr=rep_arr,
            )
            logger.debug("Trivial supercell with repetition: (%d, %d, %d)", *rep_arr)
        else:
            # Choose the generalized pathway.
            # Pre-calculate the inverse of the cell for faster wrapping of the positions.
            cell = template.get_cell()
            cell_T_inv = np.linalg.inv(cell.T)
            wrap_fnc = functools.partial(
                self._wrap_four_vectors_general, unique=unique, cell=cell, cell_T_inv=cell_T_inv
            )
            logger.debug("Non-trivial supercell, will wrap using cartesian coordinates")
        return wrap_fnc

    def _wrap_four_vectors_trivial(
        self,
        translation_vector: FourVector,
        unique_xyz: np.ndarray,
        sublattices: List[int],
        rep_arr: np.ndarray,
    ) -> Iterator[FourVector]:
        """Wrap FourVectors using the trivial shift+modulo operation"""
        # pylint: disable=no-self-use
        # We use the .tolist() method, faster to iterate the Python list than the NumPy array
        # for building the subsequent FourVectors.
        translated = np.mod(unique_xyz + translation_vector.xyz_array, rep_arr).tolist()
        for xyz, s in zip(translated, sublattices):
            yield FourVector(*xyz, s)

    def _wrap_four_vectors_general(
        self,
        translation_vector: FourVector,
        unique: Sequence[FourVector],
        cell: np.ndarray,
        cell_T_inv: np.ndarray = None,
    ) -> Iterator[FourVector]:
        """Generalized FourVector wrapping function."""
        # Translate the (x, y, z) components of the unique four-vectors
        # by this atom's (x, y, z) four-vector component
        translated_unique = [u.shift_xyz(translation_vector) for u in unique]
        # Find the new Cartesian coordinates of the translated FourVectors,
        # and wrap them back into the cell
        cartesian = self.generator.to_cartesian(*translated_unique)
        cartesian = tools.wrap_positions_3d(cartesian, cell, cell_T_inv=cell_T_inv)

        # Re-translate the wrapped-Cartesian coordinates of the unique four-vectors
        # into a four-vector representation (with a generator expression)
        sublattices = [fv.sublattice for fv in translated_unique]
        return self.generator.many_to_four_vector(cartesian, sublattices)

    def build_all(
        self,
        template_atoms: ase.Atoms,
        max_cluster_dia: Sequence[float],
        index_by_sublattice: List[List[int]],
    ) -> Tuple[ClusterList, TransMatrix]:
        """Create a ClusterList and a TransMatrix object, and calculate the norm factors."""
        # Ensure that we have built the clusters for the cutoff
        self.build(max_cluster_dia)

        at_cpy = template_atoms.copy()
        # Build the cluster list and translation matrix.
        cluster_list = self.cluster_list_for_template(at_cpy)
        trans_matrix = self.translation_matrix(at_cpy)

        # Finally calculate the norm factors, and insert them into the new cluster list
        _set_norm_factors(
            index_by_sublattice,
            trans_matrix,
            cluster_list,
        )
        return cluster_list, trans_matrix


def _set_norm_factors(
    index_by_sublattice: List[List[int]],
    trans_matrix: TransMatrix,
    cluster_list: ClusterList,
) -> None:
    """Set normalization factor for each cluster. Will mutate the ClusterList.

    The normalization factor only kicks in when the cell is too small and
    thus, include self-interactions. This methods corrects the impact of
    self-interactions.
    """
    symm_group = np.zeros(trans_matrix.n_sites, dtype=np.uintc)
    for num, group in enumerate(index_by_sublattice):
        symm_group[group] = num

    for cluster in cluster_list:
        fig_keys = list(set(cluster.get_all_figure_keys()))
        num_occ = {}
        for key in fig_keys:
            num_occ[key] = cluster_list.num_occ_figure(key, cluster.name, symm_group, trans_matrix)
        num_fig_occ = cluster.num_fig_occurences
        norm_factors = {}
        for key in fig_keys:
            tot_num = num_occ[key]
            num_unique = len(set(key.split("-")))
            norm_factors[key] = float(tot_num) / (num_unique * num_fig_occ[key])

        norm_factor_list = []
        for fig in cluster.indices:
            key = cluster.get_figure_key(fig)
            norm_factor_list.append(norm_factors[key])
        assert len(norm_factor_list) == len(cluster.indices)
        cluster.info["normalization_factor"] = norm_factor_list


class _CacheChecker:
    """Helper class to check if the
    cluster manager has already been built
    using a given set of settings.
    """

    def __init__(self):
        self.max_cluster_dia = None

    def requires_build(self, max_cluster_dia: Sequence[float]) -> bool:
        """Check if a given set of 'max_size'
        and 'max_cluster_dia' has previously been
        used to build the clusters"""
        # We don't have anything cached yet
        if self.max_cluster_dia is None:
            return True

        # Check if the parameters match
        return not np.array_equal(self.max_cluster_dia, max_cluster_dia)

    def set_cache(self, max_cluster_dia: Sequence[float]):
        # Ensure we set a copy, so no external
        # mutations affect the cache
        self.max_cluster_dia = deepcopy(max_cluster_dia)
