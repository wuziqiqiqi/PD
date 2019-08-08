from ase.build import bulk
from ase.spacegroup import crystal
from clease.trans_matrix_constructor import TransMatrixConstructor
from clease.tools import wrap_and_sort_by_position, index_by_position
import unittest


def brute_force_tm_construction(ref_indices, index_by_basis, atoms):

    tm = [[] for i in range(len(atoms))]
    for ref, indices in zip(ref_indices, index_by_basis):
        for i in indices:
            vec = atoms.get_distance(i, ref, vector=True)
            shifted = atoms.copy()
            shifted.translate(vec)
            shifted.wrap()
            tm[i] = index_by_position(shifted)
    return tm


def check_sparse_dense(sparse, dense):
    for i, row in enumerate(dense):
        sp_row = sparse[i]
        for col, val in sp_row.items():
            if val != row[col]:
                return False
    return True


class TestTransMatrixConstructor(unittest.TestCase):
    def test_fcc(self):
        atoms = bulk("Al", crystalstructure="fcc")*(4, 4, 4)
        atoms = wrap_and_sort_by_position(atoms)

        tm_constructor = TransMatrixConstructor(atoms, 5.0)

        ref_indx = [0]
        indx_by_basis = [list(range(len(atoms)))]
        tm_brute = brute_force_tm_construction(ref_indx, indx_by_basis, atoms)
        symm_group = [0]*len(atoms)

        tm_fast = tm_constructor.construct(ref_indx, symm_group)
        self.assertTrue(check_sparse_dense(tm_fast, tm_brute))

    def test_two_basis(self):
        atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.0)
        atoms[0].tag = 0
        atoms[1].tag = 1
        atoms = atoms*(3, 4, 5)
        atoms = wrap_and_sort_by_position(atoms)

        symm_group = [atom.tag for atom in atoms]
        index_by_group = [[], []]
        for atom in atoms:
            index_by_group[atom.tag].append(atom.index)
        ref_indx = [min(index_by_group[0]), min(index_by_group[1])]

        tm_brute = brute_force_tm_construction(ref_indx, index_by_group, atoms)

        tm_constructor = TransMatrixConstructor(atoms, 5.5)
        tm_fast = tm_constructor.construct(ref_indx, symm_group)
        self.assertTrue(check_sparse_dense(tm_fast, tm_brute))

    def test_four_basis(self):
        unitcell = crystal(symbols=['H', 'O', 'X', 'Fe'],
                           basis=[(0., 0., 0.), (0.39, 0.14, 0.),
                                  (0.2, 0.35, 0.5), (0.22, 0.38, 0.)],
                           spacegroup=55,
                           cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                           size=[1, 1, 1])

        for atom in unitcell:
            atom.tag = atom.index

        atoms = wrap_and_sort_by_position(unitcell*(3, 3, 2))

        symm_group = [atom.tag for atom in atoms]
        index_by_group = [[] for _ in range(len(unitcell))]

        for atom in atoms:
            index_by_group[atom.tag].append(atom.index)
        ref_indx = [min(gr) for gr in index_by_group]

        tm_brute = brute_force_tm_construction(ref_indx, index_by_group, atoms)

        tm_constructor = TransMatrixConstructor(atoms, 5.0)
        tm_fast = tm_constructor.construct(ref_indx, symm_group)
        self.assertTrue(check_sparse_dense(tm_fast, tm_brute))

    def test_sp217(self):
        a = 10.553
        b = 10.553
        c = 10.553
        alpha = 90
        beta = 90
        gamma = 90
        cellpar = [a, b, c, alpha, beta, gamma]
        basis = [(0, 0, 0), (0.324, 0.324, 0.324),
                 (0.3582, 0.3582, 0.0393), (0.0954, 0.0954, 0.2725)]

        unitcell = crystal(symbols=['Al', 'Al', 'Al', 'Al'], cellpar=cellpar,
                           spacegroup=217, primitive_cell=False,
                           basis=basis)

        # Tag atoms
        for atom in unitcell:
            atom.tag = atom.index

        atoms = wrap_and_sort_by_position(unitcell*(2, 2, 2))
        symm_group = [atom.tag for atom in atoms]

        index_by_group = [[] for _ in range(len(unitcell))]

        for atom in atoms:
            index_by_group[atom.tag].append(atom.index)
        ref_indx = [min(gr) for gr in index_by_group]

        tm_brute = brute_force_tm_construction(ref_indx, index_by_group, atoms)

        tm_constructor = TransMatrixConstructor(atoms, 5.0)
        tm_fast = tm_constructor.construct(ref_indx, symm_group)
        self.assertTrue(check_sparse_dense(tm_fast, tm_brute))


def timing():
    import time
    import numpy as np
    atoms = bulk("Al")

    brute_force_time = []
    fast_time = []
    sizes = []
    for n in range(2, 15):
        print("Current size: {}".format(n))
        atoms_scaled = atoms*(n, n, n)
        atoms_scaled = wrap_and_sort_by_position(atoms_scaled)

        symm_group = [0 for _ in range(len(atoms_scaled))]
        ref_indx = [0]
        index_by_group = [list(range(len(atoms_scaled)))]

        start = time.time()
        brute_force_tm_construction(ref_indx, index_by_group, atoms_scaled)
        brute_force_time.append(time.time() - start)

        start = time.time()
        fast_tm = TransMatrixConstructor(atoms_scaled, 5.0)
        fast_tm.construct(ref_indx, symm_group)
        fast_time.append(time.time() - start)

        sizes.append(n)

    np.savetxt("trans_mat_timing.csv",
               np.vstack((sizes, brute_force_time, fast_time)).T,
               header="Size, Brute force, fast")

# timing()


if __name__ == '__main__':
    unittest.main()
