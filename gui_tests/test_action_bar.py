import unittest
from unittest.mock import patch, MagicMock
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
import os
from clease.gui.cleaseGUI import WindowFrame


class TestActionBar(unittest.TestCase):
    @patch('clease.gui.concentrationPage.App')
    @patch('clease.gui.settingsPage.App')
    @patch('ase.visualize.view')
    @patch('clease.gui.cleaseGUI.App')
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
        wf.settings.cluster_list.get_figures.assert_called_with(at)

    @patch('clease.gui.concentrationPage.ConcentrationPage.check_user_input')
    @patch('clease.gui.settingsPage.SettingsPage.check_user_input')
    @patch('clease.gui.cleaseGUI.App')
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



if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("cleaseGUILayout.kv")
    unittest.main()
