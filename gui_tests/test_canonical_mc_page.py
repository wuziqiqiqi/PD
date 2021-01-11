import unittest
from unittest.mock import patch, MagicMock
from kivy.lang import Builder
from kivy.resources import resource_add_path
from kivy.uix.textinput import TextInput
import os.path as op
import clease.gui
from clease.gui.canonical_mc_page import CanonicalMCPage


class TestCanonicalMCPage(unittest.TestCase):

    def test_load_save(self):
        page = CanonicalMCPage()

        num_fields = 0
        for item in page.walk():
            if isinstance(item, TextInput):
                item.text = 'Some random text'
                num_fields += 1

        dict_rep = page.todict()
        self.assertEqual(len(dict_rep), num_fields)

        page2 = CanonicalMCPage()
        page2.from_dict(dict_rep)

        for item in page2.walk():
            if isinstance(item, TextInput):
                self.assertEqual(item.text, 'Some random text')

    def test_help_buttons_appear(self):
        page = CanonicalMCPage()

        for item in page.walk():
            if hasattr(item, 'text'):
                if item.text == '?':
                    self.assertIsNone(page._pop_up)
                    item.dispatch('on_release')
                    self.assertIsNotNone(page._pop_up)
                    page._pop_up.content.ids.closeButton.dispatch('on_release')
                    self.assertIsNone(page._pop_up)


if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("clease_gui.kv")
    unittest.main()
