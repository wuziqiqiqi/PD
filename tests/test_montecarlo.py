import os
import unittest
from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo
from clease.montecarlo.observers import CorrelationFunctionObserver
from clease.montecarlo.observers import Snapshot
from clease.montecarlo.observers import EnergyEvolution
from clease.montecarlo.observers import SiteOrderParameter
from clease import Concentration, CEBulk, CorrFunction


def get_example_mc_system(db_name):
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
    return atoms


class TestMonteCarlo(unittest.TestCase):
    def test_run(self):
        db_name = 'mc_test_aucu.db'
        atoms = get_example_mc_system(db_name)        

        # Insert a few elements
        for i in range(10):
            atoms[i].symbol = 'Cu'

        E = []
        for T in [1000, 500, 100]:
            mc = Montecarlo(atoms, T)
            mc.run(steps=10000)
            E.append(mc.get_thermodynamic()['energy'])

        cf = CorrFunction(atoms.get_calculator().setting)
        cf_calc = atoms.get_calculator().get_cf()
        cf_scratch = cf.get_cf(atoms)

        os.remove(db_name)
        for k, v in cf_calc.items():
            self.assertAlmostEqual(v, cf_calc[k])

        # Make sure that the energies are decreasing
        for i in range(1, len(E)):
            self.assertGreaterEqual(E[i-1], E[i])

    def test_corr_func_observer(self):
        db_name = 'test_corr_func_observer.db'
        atoms = get_example_mc_system(db_name)

        atoms[0].symbol = 'Cu'
        atoms[1].symbol = 'Cu'

        mc = Montecarlo(atoms, 600)
        obs = CorrelationFunctionObserver(atoms.get_calculator())
        mc.attach(obs, interval=1)
        mc.run(steps=1000)
        thermo = mc.get_thermodynamic()
        avg = obs.get_averages()
        os.remove(db_name)
        self.assertEqual(obs.counter, 1001)

        cf_keys = atoms.get_calculator().get_cf().keys()

        for k in cf_keys:
            self.assertTrue('cf_' + k in thermo.keys())

    def test_snapshot(self):
        db_name = 'test_snapshot.db'
        atoms = get_example_mc_system(db_name)

        atoms[0].symbol = 'Cu'
        atoms[1].symbol = 'Cu'

        traj = 'snapshot.traj'
        obs = Snapshot(trajfile=traj, atoms=atoms)

        mc = Montecarlo(atoms, 600)
        mc.attach(obs, interval=100)
        mc.run(steps=1000)
        os.remove(db_name)
        self.assertEqual(len(obs.traj), 10)
        os.remove(traj)

    def test_energy_evolution(self):
        db_name = 'test_energy_evolution.db'

        atoms = get_example_mc_system(db_name)
        atoms[0].symbol = 'Cu'
        atoms[1].symbol = 'Cu'

        obs = EnergyEvolution(atoms.get_calculator())

        mc = Montecarlo(atoms, 600)
        mc.attach(obs, interval=50)
        mc.run(steps=1000)
        fname = 'energy_evol.csv'

        # Just confirm that the save function works
        obs.save(fname=fname)
        os.remove(fname)

        # Check the number of energy values
        os.remove(db_name)
        self.assertEqual(20, len(obs.energies))

    def test_site_order_parameter(self):
        db_name = 'test_site_order.db'
        atoms = get_example_mc_system(db_name)

        atoms[0].symbol = 'Cu'
        atoms[1].symbol = 'Cu'
        atoms[2].symbol = 'Cu'

        obs = SiteOrderParameter(atoms)
        mc = Montecarlo(atoms, 600)
        mc.attach(obs)
        mc.run(steps=1000)
        avg = obs.get_averages()
        self.assertLessEqual(avg['site_order_average'], 6.0)

        thermo = mc.get_thermodynamic()

        os.remove(db_name)
        self.assertTrue('site_order_average' in thermo.keys())
        self.assertTrue('site_order_std' in thermo.keys())

if __name__ == '__main__':
    unittest.main()
