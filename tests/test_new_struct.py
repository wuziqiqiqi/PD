import unittest
from unittest.mock import MagicMock, patch
from clease import NewStructures, CEBulk
from clease import Concentration, ClusterExpansionSetting
from ase.io.trajectory import TrajectoryWriter
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.db import connect
from random import choice
import os
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


class CorrFuncPlaceholder(object):
    def get_cf(self, atoms):
        return {'c0': 0.0}


class BfSchemePlaceholder(object):
    name = 'basis_func_type'


class TestNewStruct(unittest.TestCase):
    def test_insert_structures(self):
        settings_mock = MagicMock(spec=ClusterExpansionSetting)
        settings_mock.db_name = 'test_insert_structures.db'
        settings_mock.basis_func_type = BfSchemePlaceholder()

        # Mock several methods
        new_struct = NewStructures(settings_mock)
        new_struct._exists_in_db = MagicMock(return_value=False)
        new_struct._get_formula_unit = MagicMock(return_value='AuCu')
        new_struct.corrfunc = CorrFuncPlaceholder()
        new_struct._get_kvp = MagicMock(return_value={'name': 'name'})

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

        # Test when both initial and final is given
        new_struct.insert_structures(traj_init=traj_in, traj_final=traj_final)

        # Test when only initial is given
        new_struct.insert_structures(traj_init=traj_in)
        traj_in_obj.close()
        traj_final_obj.close()
        os.remove(traj_in)
        os.remove(traj_final)

        # Run some statistics
        self.assertEqual(new_struct._exists_in_db.call_count, 2*num_struct)
        self.assertEqual(new_struct._get_formula_unit.call_count, 2*num_struct)
        self.assertEqual(new_struct._get_kvp.call_count, 2*num_struct)

        # Check that final structures has a calculator
        db = connect(settings_mock.db_name)
        for row in db.select(struct_type='final'):
            self.assertEqual(row.calculator, 'emt')
            energy = row.get('energy', None)
            self.assertTrue(energy is not None)

        os.remove(settings_mock.db_name)

    def test_determine_generation_number(self):
        db_name = 'test_gen_number.db'
        settings = MagicMock(spec=ClusterExpansionSetting, db_name=db_name)
        settings.db_name = db_name
        N = 5
        new_struct = NewStructures(
            settings, generation_number=None, struct_per_gen=N)

        def insert_in_db(n, gen):
            with connect(db_name) as db:
                for _ in range(n):
                    db.write(Atoms(), gen=gen)

        db_sequence = [
            {
                'num_insert': 0,
                'insert_gen': 0,
                'expect': 0
            },
            {
                'num_insert': 2,
                'insert_gen': 0,
                'expect': 0
            },
            {
                'num_insert': 3,
                'insert_gen': 0,
                'expect': 1
            },
            {
                'num_insert': 5,
                'insert_gen': 1,
                'expect': 2
            }
        ]

        for i, action in enumerate(db_sequence):
            insert_in_db(action['num_insert'], action['insert_gen'])
            gen = new_struct._determine_gen_number()

            msg = 'Test: #{} failed'.format(i)
            msg += 'Action: {}'.format(action)
            msg += 'returned generation: {}'.format(gen)
            self.assertEqual(gen, action['expect'], msg=msg)
        os.remove(db_name)

    @patch('clease.newStruct.GSStructure')
    def test_num_generated_structures(self, gs_mock):

        conc = Concentration(basis_elements=[['Au', 'Cu']])
        db_name = 'test_struct_gen_number.db'
        atoms = bulk('Au', a=2.9, crystalstructure='sc')*(5, 5, 5)
        atoms[0].symbol = 'Cu'
        atoms[10].symbol = 'Cu'

        def get_random_structure():
            atoms = bulk('Au', a=2.9, crystalstructure='sc')*(5, 5, 5)
            for a in atoms:
                a.symbol = choice(['Au', 'Cu'])
            atoms.set_calculator(SinglePointCalculator(atoms, energy=0.0))
            return atoms, {'c1_0': 0.0}

        gs_mock.return_value.generate = get_random_structure
        gs_mock.return_value.min_energy = 0.0

        func = [
            {
                'func': NewStructures.generate_random_structures,
                'kwargs': {}
            },
            {
                'func': NewStructures.generate_gs_structure_multiple_templates,
                'kwargs': dict(num_templates=3, num_prim_cells=10,
                               init_temp=2000, final_temp=1, num_temp=1,
                               num_steps_per_temp=1, eci=None)
            },
            {
                'func': NewStructures.generate_initial_pool,
                'kwargs': {'atoms': atoms}
            },
            {
                'func': NewStructures.generate_gs_structure,
                'kwargs': dict(atoms=atoms, init_temp=2000,
                               final_temp=1, num_temp=2,
                               num_steps_per_temp=1, eci=None,
                               random_composition=True)
            },
            {
                'func': NewStructures.generate_metropolis_trajectory,
                'kwargs': dict(atoms=atoms, random_comp=False)
            },
            {
                'func': NewStructures.generate_metropolis_trajectory,
                'kwargs': dict(atoms=atoms, random_comp=True)
            },
            {
                'func': NewStructures.generate_metropolis_trajectory,
                'kwargs': dict(atoms=None, random_comp=True)
            }
        ]

        tests = [
            {
                'gen': 0,
                'struct_per_gen': 5,
                'expect_num_to_gen': 5
            },
            {
                'gen': 0,
                'struct_per_gen': 8,
                'expect_num_to_gen': 3
            },
            {
                'gen': 1,
                'struct_per_gen': 2,
                'expect_num_to_gen': 2
            }
        ]

        # # Patch the insert method such that we don't need to calculate the
        # # correlation functions etc.
        def insert_struct_patch(self, init_struct=None, final_struct=None,
                                name=None):
            atoms = bulk('Au')
            kvp = self._get_kvp(atoms, 'Au')
            db = connect(db_name)
            db.write(atoms, kvp)

        NewStructures.insert_structure = insert_struct_patch
        NewStructures._get_formula_unit = lambda self, atoms: 'Au'

        for i, f in enumerate(func):
            settings = CEBulk(conc, max_cluster_dia=3.0, a=2.9,
                              max_cluster_size=2, crystalstructure='sc',
                              db_name=db_name)
            for j, test in enumerate(tests):
                msg = 'Test #{} failed for func #{}'.format(j, i)

                new_struct = NewStructures(
                    settings, generation_number=test['gen'],
                    struct_per_gen=test['struct_per_gen'])

                num_to_gen = new_struct.num_to_gen()

                special_msg = 'Expect num in gen {}. Got: {}'.format(
                    test['expect_num_to_gen'], num_to_gen)

                self.assertEqual(test['expect_num_to_gen'],
                                 num_to_gen, msg=msg + special_msg)

                # Call the current generation method
                f['func'](new_struct, **f['kwargs'])

                num_in_gen = new_struct.num_in_gen()
                self.assertEqual(num_in_gen, test['struct_per_gen'], msg=msg)

            os.remove(db_name)


if __name__ == '__main__':
    unittest.main()
