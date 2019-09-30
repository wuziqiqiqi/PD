import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from clease import NewStructures
from clease import CEBulk
from ase.io.trajectory import TrajectoryWriter
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.db import connect
from random import choice
from clease.cluster_list import ClusterList
import os


settings_mock = MagicMock(spec=CEBulk)


class CorrFuncPlaceholder(object):
    def get_cf(self, atoms):
        return {'c0': 0.0}


class BfSchemePlaceholder(object):
    name = 'bf_scheme'


class TestNewStruct(unittest.TestCase):
    def test_insert_structures(self):
        settings_mock.db_name = 'test_insert_structures.db'
        settings_mock.bf_scheme = BfSchemePlaceholder()

        # Mock several methods
        new_struct = NewStructures(settings_mock)
        new_struct._exists_in_db = MagicMock(return_value=False)
        new_struct._get_formula_unit = MagicMock(return_value='AuCu')
        new_struct.corrfunc = CorrFuncPlaceholder()
        new_struct._get_kvp = MagicMock(return_value={'name': 'name'})

        init_atoms = []
        final_atoms = []
        symbols = ['Au', 'Cu']
        traj_in = 'initial_structures.traj'
        traj_final = 'final_structures.traj'
        traj_in_obj = TrajectoryWriter(traj_in)
        traj_final_obj = TrajectoryWriter(traj_final)

        num_struct = 10
        for i in range(num_struct):
            init = bulk('Au')*(5, 5, 5)
            for atom in init:
                init.symbol = choice(symbols)

            final = init.copy()
            calc = EMT()
            final.set_calculator(calc)
            final.get_potential_energy()
            traj_in_obj.write(init)
            traj_final_obj.write(final)

        new_struct.insert_structures(traj_init=traj_in, traj_final=traj_final)
        os.remove(traj_in)
        os.remove(traj_final)

        # Run some statistics
        self.assertEqual(new_struct._exists_in_db.call_count, num_struct)
        self.assertEqual(new_struct._get_formula_unit.call_count, num_struct)
        self.assertEqual(new_struct._get_kvp.call_count, num_struct)

        # Check that final structures has a calculator
        db = connect(settings_mock.db_name)
        for row in db.select(struct_type='final'):
            self.assertEqual(row.calculator, 'emt')
            energy = row.get('energy', None)
            self.assertTrue(energy is not None)

        os.remove(settings_mock.db_name)

if __name__ == '__main__':
    unittest.main()
