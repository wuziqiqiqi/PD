from itertools import product, chain
from copy import deepcopy
import numpy as np
from ase.geometry import wrap_positions

from .cluster_fingerprint import ClusterFingerprint
from .cluster import Cluster
from .cluster_list import ClusterList
from .cluster_generator import ClusterGenerator
from .utils import name_clusters, size

__all__ = ('ClusterManager',)


class ClusterManager:
    """
    Manager for construction of all cluster.

    Parameters:

    prim_cell: ase.Atoms
        Primitive cell
    """

    def __init__(self, prim_cell):
        self.generator = ClusterGenerator(prim_cell)
        self.clusters = ClusterList()

    def __eq__(self, other):
        return self.clusters == other.clusters and \
            self.generator == other.generator

    def build(self, max_size=4, max_cluster_dia=4.0):
        """
        Construct all clusters.

        Parameters:

        max_size: int
            Maximum number of atoms in a cluster

        max_cluster_dia: float
            Maximum distance between two atoms in a cluster
        """
        if isinstance(max_cluster_dia, float):
            max_cluster_dia = [max_cluster_dia for _ in range(max_size + 1)]
        num_lattices = range(len(self.generator.prim))
        cluster_size = range(2, max_size + 1)
        all_fps = []
        names = []
        all_clusters = []
        all_eq_sites = []
        lattices = []
        diameters = []
        sizes = []
        for latt, s in product(num_lattices, cluster_size):
            clusters, fps = self.generator.generate(s, max_cluster_dia[s], ref_lattice=latt)

            eq_sites = []
            for c in clusters:
                eq_sites.append(self.generator.equivalent_sites(c[0]))

            all_fps += fps
            all_clusters += clusters
            all_eq_sites += eq_sites
            lattices += [latt] * len(clusters)
            sizes += [s] * len(clusters)
            diameters += [2 * np.sqrt(fp[0]) for fp in fps]

        names = self._get_names(all_fps)
        # Transfer to the cluster list
        zipped = zip(names, sizes, diameters, all_fps, all_clusters, all_eq_sites, lattices)
        for n, s, d, fp, c, eq, l in zipped:
            cluster = Cluster(name=n,
                              size=s,
                              diameter=d,
                              fingerprint=fp,
                              ref_indx=-1,
                              indices=c,
                              equiv_sites=eq,
                              trans_symm_group=l)
            self.clusters.append(cluster)

        # Add singlets and empty
        for i in range(len(self.generator.prim)):
            self.clusters.append(
                Cluster(name='c1',
                        size=1,
                        diameter=0.0,
                        fingerprint=ClusterFingerprint([1.0]),
                        ref_indx=-1,
                        indices=[[[0, 0, 0, i]]],
                        equiv_sites=[],
                        trans_symm_group=i))
            self.clusters.append(
                Cluster('c0',
                        size=0,
                        diameter=0.0,
                        fingerprint=ClusterFingerprint([0.0]),
                        ref_indx=-1,
                        indices=[],
                        equiv_sites=[],
                        trans_symm_group=i))

    def _get_names(self, all_fps):
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

        sizes = set([size(fp) for fp in all_fps])
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

    def _ref_indices(self, kdtree):
        """
        Return all reference indices

        Parameters:

        kdtree: KDTree or cKDTree
            A KDtree representation of all atomic positions
        """
        ref_indices = []
        for i in range(len(self.generator.prim)):
            d, i = kdtree.query(self.generator.cartesian([0, 0, 0, i]))
            ref_indices.append(i)
        return ref_indices

    def info_for_template(self, template):
        """
        Specialise the cluster information to a template

        Parameter:

        template: ase.Atoms
            Atoms object representing the simulation cell
        """
        unique = self.unique_four_vectors()
        lut = self.fourvec_to_indx(template, unique)
        ref_indices = [lut[(0, 0, 0, i)] for i in range(self.generator.num_sub_lattices)]

        cluster_int = deepcopy(self.clusters)
        for cluster in cluster_int:
            if cluster.size == 0:
                cluster.ref_indx = int(ref_indices[cluster.group])
                cluster.indices = []
            elif cluster.size == 1:
                cluster.ref_indx = int(ref_indices[cluster.group])
                cluster.indices = []
            else:
                cluster.indices = self.generator.to_atom_index(cluster.indices, lut)
                cluster.ref_indx = int(ref_indices[cluster.group])
        return cluster_int

    def unique_four_vectors(self):
        """
        Return a list with all unique 4-vectors consituting
        the clusters
        """
        unique = set()
        for c in self.clusters:
            if c.size == 0:
                continue
            flat = list(chain(*c.indices))
            indices = list(map(tuple, flat))
            unique.update(indices)
        return unique

    def get_figures(self):
        """
        Return a list of atoms object representing the clusters
        """
        return self.clusters.get_figures(self.generator)

    def create_four_vector_lut(self, template):
        """
        Construct a lookup table (LUT) for the index in template given the
        wrapped vector

        Parameter:

        template: Atoms
            Atoms object to use when creating the lookup table (LUT)
        """
        lut = {}
        pos = template.get_positions().copy()
        for i in range(pos.shape[0]):
            if template[i].tag >= self.generator.num_sub_lattices:
                continue
            vec = self.generator.get_four_vector(pos[i, :], template[i].tag)
            lut[tuple(vec)] = i
        return lut

    def fourvec_to_indx(self, template, unique):
        cell = template.get_cell()
        pos = np.zeros((len(unique), 3))
        for i, u in enumerate(unique):
            cart_u = self.generator.cartesian(u)
            pos[i, :] = cart_u

        pos = wrap_positions(pos, cell)
        unique_indices = []
        for i in range(pos.shape[0]):
            diff_sq = np.sum((pos[i, :] - template.get_positions())**2, axis=1)
            unique_indices.append(np.argmin(diff_sq))
        return dict(zip(unique, unique_indices))

    def translation_matrix(self, template):
        """
        Construct the translation matrix

        Parameter:

        template: ase.Atoms
            Atoms object representing the simulation cell
        """
        trans_mat = []
        unique = self.unique_four_vectors()
        cell = template.get_cell()

        unique_indx = self.fourvec_to_indx(template, unique)

        lut = self.create_four_vector_lut(template)
        indices = [0 for _ in range(len(unique))]
        cartesian = np.zeros((len(unique), 3))

        # Make a copy of the positions to avoid atoms being translated
        tmp_pos = template.get_positions().copy()
        unique_npy = np.array(list(unique))
        translated_unique = np.zeros_like(unique_npy)
        for atom in template:
            if atom.tag >= self.generator.num_sub_lattices:
                trans_mat.append({})
                continue

            vec = self.generator.get_four_vector(tmp_pos[atom.index, :], atom.tag)

            translated_unique[:, :3] = unique_npy[:, :3] + vec[:3]
            translated_unique[:, 3] = unique_npy[:, 3]
            cartesian = self.generator.cartesian(translated_unique)
            cartesian = wrap_positions(cartesian, cell)

            four_vecs = self.generator.get_four_vector(cartesian, translated_unique[:, 3])
            N = four_vecs.shape[0]
            indices = [lut[tuple(four_vecs[i, :])] for i in range(N)]

            trans_mat.append(dict(zip([int(unique_indx[u]) for u in unique], indices)))
        return trans_mat
