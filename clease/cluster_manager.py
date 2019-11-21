from clease.cluster_generator import ClusterGenerator
from clease.cluster_list import ClusterList
from clease.cluster import Cluster
from itertools import product, chain
from clease.name_clusters import name_clusters, size
from scipy.spatial import KDTree
from copy import deepcopy
import numpy as np
from clease.tools import flatten
from ase.geometry import wrap_positions
from clease.cluster_fingerprint import ClusterFingerprint


class ClusterManager(object):
    """
    Manager for construction of all cluster.

    Parameters:

    prim_cell: ase.Atoms
        Primitive cell
    """
    def __init__(self, prim_cell):
        self.generator = ClusterGenerator(prim_cell)
        self.clusters = ClusterList()

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
            max_cluster_dia = [max_cluster_dia for _ in range(max_size+1)]
        num_lattices = range(len(self.generator.prim))
        cluster_size = range(2, max_size+1)
        all_fps = []
        names = []
        all_clusters = []
        all_eq_sites = []
        lattices = []
        diameters = []
        sizes = []
        for l, s in product(num_lattices, cluster_size):
            clusters, fps = self.generator.generate(
                s, max_cluster_dia[s], ref_lattice=l)

            eq_sites = []
            for c in clusters:
                eq_sites.append(self.generator.equivalent_sites(c[0]))

            all_fps += fps
            all_clusters += clusters
            all_eq_sites += eq_sites
            lattices += [l]*len(clusters)
            sizes += [s]*len(clusters)
            diameters += [2*np.sqrt(fp[0]) for fp in fps]

        names = self._get_names(all_fps)
        # Transfer to the cluster list
        zipped = zip(names, sizes, diameters, all_fps, all_clusters,
                     all_eq_sites, lattices)
        for n, s, d, fp, c, eq, l in zipped:
            cluster = Cluster(name=n, size=s, diameter=d, fingerprint=fp,
                              ref_indx=-1, indices=c, equiv_sites=eq,
                              trans_symm_group=l)
            self.clusters.append(cluster)

        # Add singlets and empty
        for i in range(len(self.generator.prim)):
            self.clusters.append(Cluster(name='c1', size=1, diameter=0.0,
                                 fingerprint=ClusterFingerprint([1.0]),
                                 ref_indx=-1, indices=[[[0, 0, 0, i]]],
                                 equiv_sites=[], trans_symm_group=i))
            self.clusters.append(Cluster('c0', size=0, diameter=0.0,
                                         fingerprint=ClusterFingerprint([0.0]),
                                         ref_indx=-1, indices=[],
                                         equiv_sites=[], trans_symm_group=i))

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
        kdtree = KDTree(template.get_positions())
        cluster_int = deepcopy(self.clusters)
        ref_indices = self._ref_indices(kdtree)
        for cluster in cluster_int:
            if cluster.size == 0:
                cluster.ref_indx = int(ref_indices[cluster.group])
                cluster.indices = []
            elif cluster.size == 1:
                cluster.ref_indx = int(ref_indices[cluster.group])
                cluster.indices = []
            else:
                cluster.indices = self.generator.to_atom_index(
                    cluster.indices, template, kdtree=kdtree)
                cluster.ref_indx = int(ref_indices[cluster.group])

                # Convert from numpy int to regular int
                cluster.indices = [[int(x) for x in fig]
                                   for fig in cluster.indices]
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

    def translation_matrix(self, template):
        """
        Construct the translation matrix

        Parameter:

        template: ase.Atoms
            Atoms object representing the simulation cell
        """
        trans_mat = []
        kdtree = KDTree(template.get_positions())
        unique = self.unique_four_vectors()

        unique_indx = {}
        cell = template.get_cell()
        pos = np.zeros((len(unique), 3))
        for i, u in enumerate(unique):
            cart_u = self.generator.cartesian(u)
            pos[i, :] = cart_u

        pos = wrap_positions(pos, cell)
        _, i = kdtree.query(pos)
        unique_indx = dict(zip(unique, i.tolist()))

        vec2 = np.zeros(4, dtype=int)
        for atom in template:
            vec = self.generator.get_four_vector(atom.position, atom.tag)
            for i, u in enumerate(unique):
                vec2[:3] = vec[:3] + u[:3]
                vec2[3] = u[3]
                cartesian = self.generator.cartesian(vec2)
                pos[i, :] = cartesian
            pos = wrap_positions(pos, cell)
            _, j = kdtree.query(pos)
            trans_mat.append(dict(zip([int(unique_indx[u]) for u in unique],
                                      j.astype(int).tolist())))
        return trans_mat
