import unittest
from unittest.mock import patch
from clease.gui.newStructPage import NewStructPage
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
import clease.gui
import json
import os


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
    def test_exhautive_gui(self, app_mock, new_struct_mock):
        page = NewStructPage()

        # Change to Exhaustive-search
        page.ids.newStructTypeSpinner.text = 'Exhaustive Ground-state search'

        # Check active fields
        ids = page.ids
        self.assertFalse(ids.tempMaxInput.disabled)
        self.assertFalse(ids.tempMinInput.disabled)
        self.assertFalse(ids.numSweepsInput.disabled)
        self.assertFalse(ids.eciFileInput.disabled)
        self.assertFalse(ids.randomizeCompositionSpinner.disabled)
        self.assertFalse(ids.loadECIFile.disabled)
        self.assertFalse(ids.numTemplateInput.disabled)
        self.assertFalse(ids.numPrimCellsInput.disabled)
        self.assertTrue(ids.loadTemplateAtoms.disabled)
        self.assertTrue(ids.templateAtomsInput.disabled)

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

    def run(self, app):
        self.load_pop_ups(app)

if __name__ == '__main__':
    # Load the layout for the new struct page
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("newStructPageLayout.kv")
    unittest.main()
