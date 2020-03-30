import unittest
from clease import CorrFunction, Concentration, CEBulk
from clease.calculator import Clease
from clease.calculator import CleaseVolDep
from clease.tools import wrap_and_sort_by_position
from copy import deepcopy
import numpy as np
import os


def get_random_eci(setting):
    """
    Return a set of random ECIs
    """
    cfs = CorrFunction(setting).get_cf(setting.atoms)
    ecis = {k: np.random.rand() for k in cfs.keys()}
    return ecis


def get_LiVX(db_name):
    basis_elements = [['Li', 'X', 'V'], ['X', 'Li', 'V']]
    concentration = Concentration(basis_elements=basis_elements)
    setting = CEBulk(crystalstructure='rocksalt',
                     a=4.05,
                     size=[3, 3, 3],
                     concentration=concentration,
                     db_name=db_name,
                     max_cluster_size=3,
                     max_cluster_dia=[4.0, 4.0])
    return setting


class TestVolDepCalc(unittest.TestCase):
    def test_consistency(self):
        db_name = "test_consistency.db"
        setting = get_LiVX(db_name)

        atoms1 = setting.atoms.copy()
        atoms2 = atoms1.copy()
        atoms1 = wrap_and_sort_by_position(atoms1)
        atoms2 = wrap_and_sort_by_position(atoms2)

        eci = get_random_eci(setting)
        calc1 = Clease(deepcopy(setting), eci)

        eci_vol = {k + '_V0': v for k, v in eci.items()}
        vol_coeff = deepcopy(eci)
        calc2 = CleaseVolDep(deepcopy(setting), eci_vol, vol_coeff)

        atoms1.set_calculator(calc1)
        atoms2.set_calculator(calc2)

        swaps = [
            {
                'idx': 0,
                'symbol': 'X'
            },
            {
                'idx': 12,
                'symbol': 'V',
            },
            {
                'idx': 9,
                'symbol': 'V'
            }
        ]

        os.remove(db_name)
        orig_atoms1 = atoms1.copy()
        orig_atoms2 = atoms2.copy()
        calc_type = ['get_pot_en', 'get_en_given_change']

        for c in calc_type:
            atoms1.numbers = orig_atoms1.numbers
            atoms2.numbers = orig_atoms2.numbers

            # Run one energy evaluate to initialize the corr functions
            atoms1.get_potential_energy()
            atoms2.get_potential_energy()
            for s in swaps:
                change = [(s['idx'], atoms1[s['idx']].symbol, s['symbol'])]
                atoms1[s['idx']].symbol = s['symbol']
                atoms2[s['idx']].symbol = s['symbol']

                if c == 'get_pot_en':
                    E1 = atoms1.get_potential_energy()
                    E2 = atoms2.get_potential_energy()
                elif c == 'get_en_given_changet':
                    E1 = calc1.get_energy_given_change(change)
                    E2 = calc2.get_energy_given_change(change)
                self.assertAlmostEqual(E1, E2)

    def test_vol_pressure_bulk_mod(self):
        db_name = 'test_pressure.db'
        settings = get_LiVX(db_name)
        atoms = settings.atoms.copy()
        eci_keys = list(get_random_eci(settings).keys())[:2]
        eci = {
            eci_keys[0] + '_V0': 2.0,
            eci_keys[0] + '_V1': -1.0,
            eci_keys[0] + '_V2': 0.5,
            eci_keys[1] + '_V0': 3.0,
            eci_keys[1] + '_V1': 1.0,
            eci_keys[1] + '_V2': -0.5,
        }

        vol_coeff = {
            eci_keys[0]: 2.0,
            eci_keys[1]: -1.0
        }

        # Use fixed correlation functions such that we can test the result
        cf = {
            eci_keys[0]: 1.0,
            eci_keys[1]: 1.0
        }

        calc = CleaseVolDep(settings, eci, vol_coeff)
        atoms.set_calculator(calc)

        os.remove(db_name)

        vol = calc.get_volume(cf)
        expect_vol = 1.0
        self.assertAlmostEqual(vol, expect_vol)

        P = calc.get_pressure(cf)
        expect_P = 0.0
        self.assertAlmostEqual(P, expect_P)

        B = calc.get_bulk_modulus(cf)
        expect_B = 0.0
        self.assertAlmostEqual(B, expect_B)

    def test_update_eci(self):
        db_name = 'test_update_eci.db'
        settings = get_LiVX(db_name)
        vol_coeff = get_random_eci(settings)
        eci_vol = {k + '_V1': v for k, v in vol_coeff.items()}

        atoms = settings.atoms.copy()
        calc = CleaseVolDep(settings, eci_vol, vol_coeff)
        atoms.set_calculator(calc)

        # Mimic that we wish to change the chemical potential
        chem_pot = {'c1_0': -2.0, 'c1_1': 0.5}
        current_eci = deepcopy(calc.eci)
        for k, v in chem_pot.items():
            current_eci[k] += v
        calc.update_eci(current_eci)

        # Check that the changes got propagated to the volume dependent
        # ECIs
        for k, v in chem_pot.items():
            vol_key = k + '_V0'
            self.assertAlmostEqual(calc.eci_with_vol[vol_key], v)


if __name__ == '__main__':
    unittest.main()
