import unittest
from clease.atoms_manager import AtomsManager
from ase.build import bulk


class TestAtomsManager(unittest.TestCase):
    def test_binary(self):
        atoms = bulk('Au')*(3, 3, 3)

        # Tag even indices with 0 and odd indices with 1
        for atom in atoms:
            atom.tag = atom.index % 2

        manager = AtomsManager(atoms)
        ind_by_tag = manager.index_by_tag()
        self.assertTrue(all(map(lambda x: x % 2 == 0, ind_by_tag[0])))
        self.assertTrue(all(map(lambda x: x % 2 == 1, ind_by_tag[1])))

    def test_group_by_symbol_single(self):
        atoms = bulk('Au')*(3, 3, 3)

        for atom in atoms:
            if atom.index % 3 == 1:
                atom.symbol = 'Cu'
            elif atom.index % 3 == 2:
                atom.symbol = 'X'

        manager = AtomsManager(atoms)
        ind_by_basis = manager.index_by_symbol(['Au', 'Cu', 'X'])

        for i, items in enumerate(ind_by_basis):
            self.assertTrue(all(map(lambda x: x % 3 == i, items)))

    def test_group_by_symbol_grouped(self):
        atoms = bulk('Au')*(3, 4, 5)

        for atom in atoms:
            if atom.index % 4 == 1:
                atom.symbol = 'Cu'
            elif atom.index % 4 == 2:
                atom.symbol = 'X'
            elif atom.index % 4 == 3:
                atom.symbol = 'Ag'

        manager = AtomsManager(atoms)
        ind_by_basis = manager.index_by_symbol(['Au', ['Cu', 'X'], 'Ag'])

        self.assertTrue(all(map(lambda x: x % 4 == 0, ind_by_basis[0])))
        self.assertTrue(all(map(lambda x: x % 4 == 1 or x % 4 == 2,
                            ind_by_basis[1])))
        self.assertTrue(all(map(lambda x: x % 4 == 3, ind_by_basis[2])))


if __name__ == '__main__':
    unittest.main()
