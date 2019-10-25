import unittest
from unittest.mock import patch, MagicMock
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
from clease.gui.mc_header import MCHeader
from clease.gui.constants import SCREEN_TRANSLATIONS
from ase.atoms import Atoms


class TestMCHeader(unittest.TestCase):
    def test_change_screen(self):
        page = MCHeader()
        items = page.ids.pageSpinner.values
        sm = page.ids.sm

        for name in items:
            page.mc_type = name

            self.assertEqual(sm.current, SCREEN_TRANSLATIONS[name])


if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("cleaseGUILayout.kv")
    unittest.main()
