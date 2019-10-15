import unittest
from unittest.mock import patch, MagicMock
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
from clease.gui.cleaseGUI import WindowFrame
import time


class TestActionBar(unittest.TestCase):
    @patch('clease.gui.concentrationPage.App')
    @patch('ase.visualize.view')
    @patch('clease.gui.cleaseGUI.App')
    def test_view_clusters(self, app_mock, view_mock, _):
        wf = WindowFrame()
        app_mock.get_running_app.return_value = MagicMock(root=wf)

        wf.view_clusters()

        # Check status message
        msg = "Settings should be applied/loaded before viewing clusters."
        self.assertEqual(wf.ids.status.text, msg)

        # Change settings such that it is not None
        at = MagicMock()
        wf.settings = MagicMock(atoms=at)
        wf.view_clusters()
        wf.settings.cluster_list.get_figures.assert_called_with(at)

if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("cleaseGUILayout.kv")
    unittest.main()
