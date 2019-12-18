import unittest
from clease.gui.newStructPage import NewStructPage, InsertStructureCB
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
import clease.gui
import json
import os
from unittest.mock import patch, MagicMock
from ase.build import bulk
from ase.io.trajectory import TrajectoryWriter
from ase.io import write


class NewStructPageTest(unittest.TestCase):
    def load_pop_ups(self, app):
        screen = app.root.ids.sm.get_screen('NewStruct')

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadInitStruct.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertEqual(screen._pop_up.title, "Load initial structure")
        screen.dismiss_popup()

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadFinalStruct.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertEqual(screen._pop_up.title, "Load final structure")
        screen.dismiss_popup()

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadECIFile.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertEqual(screen._pop_up.title, "Load ECI filename")
        screen.dismiss_popup()

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadTemplateAtoms.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertEqual(screen._pop_up.title, "Load template atoms")
        screen.dismiss_popup()

    @patch('clease.NewStructures')
    @patch('clease.gui.newStructPage.App')
    def test_exhaustive_gui(self, app_mock, new_struct_mock):
        page = NewStructPage()

        # Change to Exhaustive-search
        gen_type = 'Ground-state structure (variable template)'
        page.ids.newStructTypeSpinner.text = gen_type

        # Check active fields
        ids = page.ids
        self.assertFalse(ids.tempMaxInput.disabled)
        self.assertFalse(ids.tempMinInput.disabled)
        self.assertFalse(ids.numSweepsInput.disabled)
        self.assertFalse(ids.eciFileInput.disabled)
        self.assertFalse(ids.loadECIFile.disabled)
        self.assertFalse(ids.numTemplateInput.disabled)
        self.assertFalse(ids.numPrimCellsInput.disabled)
        self.assertTrue(ids.loadTemplateAtoms.disabled)
        self.assertTrue(ids.templateAtomsInput.disabled)
        self.assertTrue(ids.randomizeCompositionSpinner.disabled)

        # Create an ECI file
        ecis = {'c0': 1.0}
        eci_file = 'example_eci_new_struct_page.json'
        with open(eci_file, 'w') as out:
            json.dump(ecis, out)

        # Update ECI file
        page.ids.eciFileInput.text = eci_file

        # Try to generate a structure
        page.ids.generateButton.dispatch('on_release')

        # Make sure functions where called correctly
        method = new_struct_mock().generate_gs_structure_multiple_templates
        os.remove(eci_file)
        self.assertEqual(method.call_count, 1)

    @patch('clease.gui.newStructPage.InsertStructureCB')
    @patch('clease.gui.newStructPage.App')
    @patch('clease.NewStructures')
    def test_import_structures(self, new_struct_mock, kivy_mock, cb_mock):
        init_file = 'initial_structure.traj'
        final_file = 'final_structure.traj'
        traj_init = TrajectoryWriter(init_file)
        traj_final = TrajectoryWriter(final_file)

        for _ in range(2):
            atoms = bulk('Au')
            traj_init.write(atoms)
            traj_final.write(atoms)

        page = NewStructPage()
        page.ids.initStructInput = MagicMock(text=init_file)
        page.ids.finalStructInput = MagicMock(text=final_file)
        page.import_structures()

        # Check the argument passed to the cb_mock class
        cb_mock.assert_called_once_with(
            kivy_mock.get_running_app().root.ids.status)

        # Make sure the method is called exactly once
        method = new_struct_mock().insert_structures

        # Make sure that insert_structures was called with the right arguments
        method.assert_called_once_with(
            traj_init=init_file, traj_final=final_file, cb=cb_mock())

        # Check that it is called correctly also when only the initial
        # structures are passed
        page.ids.finalStructInput.text = ''
        cb_mock.reset_mock()
        method.reset_mock()
        page.import_structures()

        cb_mock.assert_called_once_with(
            kivy_mock.get_running_app().root.ids.status)
        method.assert_called_once_with(traj_init=init_file, cb=cb_mock())


        del traj_init
        del traj_final
        os.remove(init_file)
        os.remove(final_file)

        # Try to read xyz files
        init_file = 'initial_structure.xyz'
        final_file = 'final_structure.xyz'
        atoms = bulk('Au')
        write(init_file, atoms)
        write(final_file, atoms)
        page.ids.initStructInput = MagicMock(text=init_file)
        page.ids.finalStructInput = MagicMock(text=final_file)
        page.import_structures()

        method = new_struct_mock().insert_structure
        self.assertEqual(method.call_count, 1)
        os.remove(init_file)
        os.remove(final_file)

    def test_insert_structure_cb(self):
        status = MagicMock()
        insCB = InsertStructureCB(status)
        insCB(2, 10)
        self.assertEqual(status.text, 'Inserted 2 of 10 structures')

    def run_with_app(self, app):
        self.load_pop_ups(app)

    @patch('clease.gui.newStructPage.App')
    @patch('clease.NewStructures')
    def test_missing_eci_file(self, ns_mock, app_mock):
        # Set up the Mock such that we can check the error message
        status = MagicMock(text='')
        ids = MagicMock(status=status)
        root = MagicMock(ids=ids)
        app_mock.get_running_app = MagicMock(return_value=MagicMock(root=root))

        page = NewStructPage()
        struct_type = 'Ground-state structure (fixed template)'
        page.ids.newStructTypeSpinner.text = struct_type

        # Try to generate a structure (when ECI file is not given)
        page.ids.generateButton.dispatch('on_release')
        self.assertEqual(status.text, 'No ECI file given')

    def test_to_from_dict(self):
        page = NewStructPage()
        page.ids.genNumberInput.text = '2'
        page.ids.structPerGenInput.text = '23'
        page.ids.initStructInput.text = 'none'
        page.ids.finalStructInput.text = 'none'
        page.ids.tempMinInput.text = '10'
        page.ids.tempMaxInput.text = '100'
        page.ids.numTempInput.text = '21'
        page.ids.numTemplateInput.text = '20'
        page.ids.numSweepsInput.text = '100'
        page.ids.eciFileInput.text = 'myecis.json'
        page.ids.templateAtomsInput.text = 'mytemplate.xyz'
        page.ids.newStructTypeSpinner.text = 'Random structure'

        dct_rep = page.to_dict()

        expect = {
            'gen_number': '2',
            'struct_per_gen': '23',
            'init_struct': 'none',
            'final_struct': 'none',
            'min_temp': '10',
            'max_temp': '100',
            'num_temps': '21',
            'num_sweeps': '100',
            'eci_file': 'myecis.json',
            'template_file': 'mytemplate.xyz',
            'generation_scheme': 'Random structure',
            'num_templates': '20'
        }
        self.assertDictEqual(dct_rep, expect)

        page2 = NewStructPage()
        page2.from_dict(expect)

        # Check that all text fields matches the first page
        txt1 = [w.text for w in page.walk() if hasattr(w, 'text')]
        txt2 = [w.text for w in page2.walk() if hasattr(w, 'text')]

        self.assertGreater(len(txt1), 0)
        self.assertEqual(txt1, txt2)


if __name__ == '__main__':
    # Load the layout for the new struct page
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("newStructPageLayout.kv")
    unittest.main()
