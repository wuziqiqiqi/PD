import unittest
from clease.montecarlo.observers import ConcentrationObserver
from clease.datastructures import SystemChange
from ase.build import bulk


class TestConcObsrever(unittest.TestCase):
    def test_update(self):
        atoms = bulk("Au") * (7, 7, 7)
        obs = ConcentrationObserver(atoms, element="Au")
        self.assertAlmostEqual(obs.current_conc, 1.0)

        changes = [
            SystemChange(0, "Au", "Cu"),
            SystemChange(1, "Au", "Cu"),
            SystemChange(2, "Au", "Cu"),
        ]
        obs(changes)
        N = len(atoms)
        self.assertAlmostEqual(obs.current_conc, 1.0 - 3.0 / N)

        changes = [SystemChange(0, "Cu", "Au")]
        obs(changes)
        self.assertAlmostEqual(obs.current_conc, 1.0 - 2.0 / N)

    def test_peak(self):
        atoms = bulk("Au") * (7, 7, 7)
        obs = ConcentrationObserver(atoms, element="Au")
        self.assertAlmostEqual(obs.current_conc, 1.0)

        changes = [
            SystemChange(0, "Au", "Cu"),
            SystemChange(1, "Au", "Cu"),
            SystemChange(2, "Au", "Cu"),
        ]
        new_conc = obs(changes, peak=True)
        self.assertAlmostEqual(new_conc, 1.0 - 3.0 / len(atoms))
        self.assertAlmostEqual(obs.current_conc, 1.0)

    def test_reset(self):
        atoms = bulk("Au") * (7, 7, 7)
        obs = ConcentrationObserver(atoms, element="Au")
        self.assertAlmostEqual(obs.current_conc, 1.0)

        changes = [
            SystemChange(0, "Au", "Cu"),
            SystemChange(1, "Au", "Cu"),
            SystemChange(2, "Au", "Cu"),
        ]
        new_conc = obs(changes)
        avg = obs.get_averages()
        expect = (1.0 + 1.0 - 3.0 / len(atoms)) * 0.5
        self.assertAlmostEqual(avg["conc_Au"], expect)

        obs.reset()
        avg = obs.get_averages()
        self.assertAlmostEqual(avg["conc_Au"], 1.0 - 3.0 / len(atoms))

    def test_none(self):
        atoms = bulk("Au") * (7, 7, 7)
        obs = ConcentrationObserver(atoms, element="Au")
        conc = obs(None)
        self.assertAlmostEqual(conc, 1.0)


if __name__ == "__main__":
    unittest.main()
