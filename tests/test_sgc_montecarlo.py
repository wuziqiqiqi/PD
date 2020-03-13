import os
import unittest
from clease.calculator import attach_calculator
from clease.montecarlo import SGCMonteCarlo
from clease import Concentration, CEBulk, CorrFunction
from clease.montecarlo.constraints import ConstrainElementInserts
from clease.montecarlo.constraints import PairConstraint
import numpy as np


class TestSGCMonteCarlo(unittest.TestCase):
    def test_run(self):
        db_name = 'sgc_test_aucu.db'
        conc = Concentration(basis_elements=[['Au', 'Cu']])
        settings = CEBulk(db_name=db_name, concentration=conc,
                          crystalstructure='fcc', a=4.0,
                          max_cluster_size=3, max_cluster_dia=[5.0, 4.1],
                          size=[2, 2, 2])

        atoms = settings.atoms.copy()*(3, 3, 3)
        cf = CorrFunction(settings)
        cf_scratch = cf.get_cf(settings.atoms)
        eci = {k: 0.0 for k, v in cf_scratch.items()}

        eci['c0'] = -1.0
        eci['c2_d0000_0_00'] = -0.2
        atoms = attach_calculator(settings, atoms=atoms, eci=eci)

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

    def test_constrain_inserts(self):
        db_name = 'test_constrain_inserts.db'
        conc = Concentration(basis_elements=[['Si', 'X'], ['O', 'C']])
        settings = CEBulk(db_name=db_name, concentration=conc,
                          crystalstructure='rocksalt', a=4.0,
                          max_cluster_size=3, max_cluster_dia=[2.51, 3.0],
                          size=[2, 2, 2])
        atoms = settings.atoms.copy()*(3, 3, 3)
        cf = CorrFunction(settings)
        cf_scratch = cf.get_cf(settings.atoms)
        eci = {k: 0.0 for k, v in cf_scratch.items()}

        eci['c0'] = -1.0
        eci['c2_d0000_0_00'] = -0.2
        atoms = attach_calculator(settings, atoms=atoms, eci=eci)

        mc = SGCMonteCarlo(atoms, 100000, symbols=['Si', 'X', 'O', 'C'])

        elem_basis = [['Si', 'X'], ['O', 'C']]
        cnst = ConstrainElementInserts(atoms, settings.index_by_basis,
                                       elem_basis)
        chem_pot = {'c1_0': 0.0, 'c1_1': -0.1, 'c1_2': 0.1}
        mc.add_constraint(cnst)
        orig_symbols = [a.symbol for a in atoms]
        mc.run(steps=1000, chem_pot=chem_pot)

        new_symb = [a.symbol for a in atoms]
        os.remove(db_name)
        # Confirm that swaps have taken place
        self.assertFalse(all(map(lambda x: x[0] == x[1],
                                 zip(orig_symbols, new_symb))))

        # Check that we only have correct entries
        for allowed, basis in zip(elem_basis, settings.index_by_basis):
            for index in basis:
                self.assertTrue(atoms[index].symbol in allowed)

    def test_pair_constraint(self):
        db_name = 'test_constrain_pair.db'
        a = 4.0
        conc = Concentration(basis_elements=[['Si', 'X']])
        settings = CEBulk(db_name=db_name, concentration=conc,
                          crystalstructure='fcc', a=a,
                          max_cluster_size=3, max_cluster_dia=[3.9, 3.0],
                          size=[2, 2, 2])

        atoms = settings.atoms.copy()*(3, 3, 3)
        cf = CorrFunction(settings)
        cf_scratch = cf.get_cf(settings.atoms)
        eci = {k: 0.0 for k, v in cf_scratch.items()}
        atoms = attach_calculator(settings, atoms=atoms, eci=eci)
        mc = SGCMonteCarlo(atoms, 100000000, symbols=['Si', 'X'])

        cluster = settings.cluster_list.get_by_name("c2_d0000_0")[0]
        cnst = PairConstraint(
            ['X', 'X'], cluster, settings.trans_matrix, atoms)
        mc.add_constraint(cnst)

        nn_dist = a/np.sqrt(2.0)
        for num in range(20):
            mc.run(10, chem_pot={'c1_0': 0.0})

            X_idx = [atom.index for atom in atoms if atom.symbol == 'X']

            if len(X_idx) <= 1:
                continue

            # Check that there are no X that are nearest neighbours
            dists = [atoms.get_distance(i1, i2) for i1 in X_idx
                     for i2 in X_idx[i1+1:]]
            self.assertTrue(all(d > 1.05*nn_dist for d in dists))
        os.remove(db_name)


if __name__ == '__main__':
    unittest.main()
