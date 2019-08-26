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
        ind_by_symbol = manager.index_by_symbol(['Au', 'Cu', 'X'])

        for i, items in enumerate(ind_by_symbol):
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
        ind_by_symbol = manager.index_by_symbol(['Au', ['Cu', 'X'], 'Ag'])

        self.assertTrue(all(map(lambda x: x % 4 == 0, ind_by_symbol[0])))
        self.assertTrue(all(map(lambda x: x % 4 == 1 or x % 4 == 2,
                                ind_by_symbol[1])))
        self.assertTrue(all(map(lambda x: x % 4 == 3, ind_by_symbol[2])))
        self.assertEqual(sorted(manager.unique_elements()),
                         ['Ag', 'Au', 'Cu', 'X'])

    def test_background_indices(self):
        atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.0)
        atoms = atoms*(5, 5, 5)

        # Chlorine sites are background indices
        basis_elements = [['Na', 'X'], ['Cl']]
        manager = AtomsManager(atoms)
        bkg_indices = manager.single_element_sites(basis_elements)

        cl_sites = [atom.index for atom in atoms if atom.symbol == 'Cl']
        self.assertEqual(sorted(bkg_indices), sorted(cl_sites))

        # Extract unique elements
        unique_elem = manager.unique_elements()
        self.assertEqual(sorted(unique_elem), ['Cl', 'Na'])

        # Try unique elements without background
        unique_elem = manager.unique_elements(ignore=['Cl'])
        self.assertEqual(sorted(unique_elem), ['Na'])

    def test_tag_by_corresponding_atom(self):
        prim_cell = bulk('Mg', crystalstructure='hcp')
        prim_cell[0].symbol = 'Mg'
        prim_cell[1].symbol = 'Zn'

        atoms = prim_cell*(2, 3, 4)
        manager = AtomsManager(atoms)
        manager.tag_indices_of_corresponding_atom(prim_cell)

        for atom in manager.atoms:
            if atom.symbol == 'Mg':
                self.assertEqual(atom.tag, 0)
            else:
                self.assertEqual(atom.tag, 1)

    def test_tag_by_corresponding_primitive_conventional(self):
        prim_cell = bulk('NaCl', crystalstructure='rocksalt', a=4.0)
        prim_cell.wrap()
        atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.0, cubic=True)
        atoms = atoms*(3, 4, 5)

        manager = AtomsManager(atoms)
        manager.tag_indices_of_corresponding_atom(prim_cell)

        for atom in manager.atoms:
            if atom.symbol == 'Na':
                self.assertEqual(atom.tag, 0)
            elif atom.symbol == 'Cl':
                self.assertEqual(atom.tag, 1)

    def test_raise_if_not_match(self):
        prim_cell = bulk('NaCl', crystalstructure='rocksalt', a=4.0)
        prim_cell.wrap()
        atoms = bulk('Mg', crystalstructure='hcp')
        atoms = atoms*(3, 4, 5)

        manager = AtomsManager(atoms)
        with self.assertRaises(ValueError):
            manager.tag_indices_of_corresponding_atom(prim_cell)


if __name__ == '__main__':
    unittest.main()
