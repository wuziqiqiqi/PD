import unittest
from ase.build import bulk
from clease.mc_trajectory_extractor import MCTrajectoryExtractor


class TestMCTrajectoryExtractor(unittest.TestCase):
    def test_is_related_by_swap(self):
        atAl = bulk("Al") * (2, 2, 2)
        atAl[0].symbol = "X"

        atAl2 = bulk("Al") * (2, 2, 2)
        atAl2[4].symbol = "X"

        atAl3 = bulk("Al") * (2, 2, 2)
        atAl3[0].symbol = "X"
        atAl3[4].symbol = "X"

        atAl4 = bulk("Al") * (2, 2, 2)
        atAl4[0].symbol = "X"
        atAl4[1].symbol = "Cu"

        tests = [
            {"atoms1": bulk("Al") * (2, 2, 2), "atoms2": bulk("Al"), "expect": False},
            {"atoms1": atAl, "atoms2": atAl2, "expect": True},
            {"atoms1": atAl, "atoms2": atAl3, "expect": False},
            {"atoms1": atAl, "atoms2": atAl4, "expect": False},
        ]

        extractor = MCTrajectoryExtractor()
        for t in tests:
            res = extractor.is_related_by_swap(t["atoms1"], t["atoms2"])
            self.assertEqual(res, t["expect"])

    def test_find_swaps(self):
        atoms = bulk("Al") * (4, 4, 4)
        atoms[0].symbol = "X"

        atoms1 = atoms.copy()
        atoms1[0].symbol = "Al"
        atoms1[10].symbol = "X"

        atoms2 = atoms.copy()
        atoms2[0].symbol = "Al"
        atoms2[3].symbol = "X"

        # Structures with two X
        atoms3 = atoms.copy()
        atoms3[10].symbol = "X"

        atoms4 = atoms.copy()
        atoms4[32].symbol = "X"

        all_atoms = [atoms, atoms1, atoms2, atoms3, atoms4]
        expect = [(0, 1), (0, 2), (1, 2), (3, 4)]
        extractor = MCTrajectoryExtractor()
        swaps = extractor.find_swaps(all_atoms)
        self.assertEqual(swaps, expect)


if __name__ == "__main__":
    unittest.main()
