import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from clease.gui.newStructPage import NewStructPage
from clease import CEBulk
from ase.build import bulk
from ase.io.trajectory import TrajectoryWriter
import os
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

    @patch('clease.gui.newStructPage.App')
    @patch('clease.NewStructures')
    def test_import_structures(self, new_struct_mock, kivy_mock):
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

        # Make sure the method is called exactly once
        method = new_struct_mock().insert_structures
        self.assertEqual(method.call_count, 1)

        # Make sure that insert_structures was called with the right arguments
        method.assert_called_with(traj_init=init_file, traj_final=final_file)
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

    def run_with_app(self, app):
        self.load_pop_ups(app)

if __name__ == '__main__':
    unittest.main()
