import unittest
from unittest.mock import patch, MagicMock
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
import os
from clease.gui.clease_gui import WindowFrame


class TestActionBar(unittest.TestCase):

    @patch('clease.gui.concentration_page.App')
    @patch('clease.gui.settings_page.App')
    @patch('ase.visualize.view')
    @patch('clease.gui.clease_gui.App')
    def test_view_clusters(self, app_mock, view_mock, app_set, _):
        wf = WindowFrame()
        app_mock.get_running_app = MagicMock(return_value=MagicMock(root=wf))
        app_set.get_running_app = MagicMock(return_value=MagicMock(root=wf))
        wf.view_clusters()

        # Check status message
        msg = "Settings should be applied/loaded before viewing clusters."
        self.assertEqual(wf.ids.status.text, msg)

        # Change settings such that it is not None
        at = MagicMock()
        wf.settings = MagicMock(atoms=at)
        wf.view_clusters()
        wf.settings.cluster_mng.get_figures.assert_called_with()

    @patch('clease.gui.concentration_page.ConcentrationPage.check_user_input')
    @patch('clease.gui.settings_page.SettingsPage.check_user_input')
    @patch('clease.gui.clease_gui.App')
    def test_save_session(self, app_mock, settings_usr_input, conc_usr_input):
        wf = WindowFrame()
        app_mock.get_running_app = MagicMock(return_value=MagicMock(root=wf))
        settings_usr_input.return_value = 0
        conc_usr_input.return_value = 0

        # Just verify that this method runs without raising errors. Currently
        # it is up to each page to check that the to_dict method works properly
        fname = 'session_save.json'
        wf.current_session_file = fname
        wf.save_session_to_current_file()
        wf.load_session('', [fname])
        os.remove(fname)

    @patch('clease.gui.clease_gui.Evaluate')
    @patch('clease.gui.clease_gui.App')
    def test_export_dataset(self, app_mock, eval_mock):
        wf = WindowFrame()
        app_mock.get_running_app = MagicMock(return_value=MagicMock(root=wf))

        wf.show_export_fit_data_dialog()

        # Confirm that the correct pop-up opens
        self.assertEqual(wf._pop_up.title, "Export Fit Data")

        fname = "demoFileOut.csv"
        content = wf._pop_up.content
        content.ids.filechooser.path = "."
        self.assertEqual(content.ids.userFilename.text, "fitData.csv")
        content.ids.userFilename.text = fname

        # Close the dialog
        content.ids.saveButton.dispatch('on_release')

        self.assertIsNone(wf._pop_up)
        eval_mock.return_value.export_dataset.assert_called_with(fname)

    @patch('clease.gui.clease_gui.App')
    def test_export_settings(self, app_mock):
        wf = WindowFrame()
        wf.settings = MagicMock()
        app_mock.get_running_app = MagicMock(return_value=MagicMock(root=wf))

        wf.show_export_settings_dialog()
        self.assertEqual(wf._pop_up.title, "Export Settings")
        fname = "settings.json"
        content = wf._pop_up.content
        content.ids.filechooser.path = "."
        self.assertEqual(content.ids.userFilename.text, 'cleaseSettings.json')
        content.ids.userFilename.text = fname

        # Close the dialog
        content.ids.saveButton.dispatch('on_release')
        self.assertIsNone(wf._pop_up)
        wf.settings.save.assert_called_with(fname)


if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("clease_gui.kv")
    unittest.main()
