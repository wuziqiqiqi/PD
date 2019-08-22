import os
import unittest
from clease.calculator import attach_calculator
from clease.montecarlo import SGCMonteCarlo
from clease import Concentration, CEBulk, CorrFunction


class TestSGCMonteCarlo(unittest.TestCase):
    def test_run(self):
        db_name = 'sgc_test_aucu.db'
        conc = Concentration(basis_elements=[['Au', 'Cu']])
        setting = CEBulk(db_name=db_name, concentration=conc,
                         crystalstructure='fcc', a=4.0,
                         max_cluster_size=3, max_cluster_dia=[5.0, 4.1],
                         size=[2, 2, 2])

        atoms = setting.atoms.copy()*(3, 3, 3)
        cf = CorrFunction(setting)
        cf_scratch = cf.get_cf(setting.atoms)
        eci = {k: 0.0 for k, v in cf_scratch.items()}

        eci['c0'] = -1.0
        eci['c2_01nn_0_00'] = -0.2
        atoms = attach_calculator(setting, atoms=atoms, eci=eci)

        E = []
        for T in [5000, 2000, 1000, 500, 100]:
            mc = SGCMonteCarlo(atoms, T, symbols=['Au', 'Cu'])
            mc.run(steps=10000, chem_pot={'c1_0': -0.02})
            E.append(mc.get_thermodynamic_quantities()['energy'])

        cf_calc = atoms.get_calculator().get_cf()
        cf_scratch = cf.get_cf(atoms)

        os.remove(db_name)
        for k, v in cf_calc.items():
            self.assertAlmostEqual(v, cf_calc[k])

        # Make sure that the energies are decreasing
        for i in range(1, len(E)):
            self.assertGreaterEqual(E[i-1], E[i])


if __name__ == '__main__':
    unittest.main()