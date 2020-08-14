import unittest
from clease.gui.conc_obs_editor import ConcObsEditor
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
import os


def close_func(text):
    return text


class TestConcObsEditor(unittest.TestCase):

    def test_load(self):
        editor = ConcObsEditor(close=close_func, default_element='Au')
        editor.ids.elementInput.text = 'Al'
        editor.backup()
        editor.ids.closeButton.dispatch('on_release')

        # Now a backup file should have been created
        editor2 = ConcObsEditor(close=close_func, default_element='Au')
        self.assertEqual(editor2.ids.elementInput.text, 'Al')
        os.remove(editor.fname)


if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("clease_gui.kv")
    unittest.main()
