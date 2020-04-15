import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from kivy.uix.textinput import TextInput
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
from clease.gui.mc_main_page import MCMainPage
from ase.build import bulk
import numpy as np


class TestMCMainPage(unittest.TestCase):
    def test_load_popup(self):
        page = MCMainPage()
        self.assertIsNone(page._pop_up)
        page.ids.loadButton.dispatch('on_release')
        self.assertIsNotNone(page._pop_up)

        # Set the selections
        page._pop_up.content.ids.filechooser.path = 'mypath/'
        page._pop_up.content.ids.filechooser.selection = ['mypath/eci.json']
        page._pop_up.content.ids.loadButton.dispatch('on_release')

        self.assertIsNone(page._pop_up)
        self.assertEqual(page.eci_file, 'mypath/eci.json')

    def test_all_help_boxes_open(self):
        page = MCMainPage()

        for item in page.walk():
            if hasattr(item, 'text'):
                if item.text == '?':
                    self.assertIsNone(page._pop_up)
                    item.dispatch('on_release')
                    self.assertIsNotNone(page._pop_up)
                    page._pop_up.content.ids.closeButton.dispatch('on_release')
                    self.assertIsNone(page._pop_up)

    def test_cell_info_update(self):
        size = 3
        atoms = bulk('Al', a=4.05, crystalstructure='fcc')*(size, size, size)

        MCMainPage.get_mc_cell = MagicMock(return_value=atoms)
        page = MCMainPage()
        page.ids.sizeInput.text = str(size)
        page.ids.sizeInput.dispatch('on_text_validate')

        lengths_ang = atoms.get_cell_lengths_and_angles()
        length_str = page.mc_cell_lengths
        remove = ['a:', 'b:', 'c:', 'Å']
        for item in remove:
            length_str = length_str.replace(item, '')
        lengths = list(map(float, length_str.split('  ')))
        self.assertTrue(np.allclose(lengths, np.floor(lengths_ang[:3])))

        ang_str = page.angle_info
        ang_str = ang_str.replace('deg', '')
        ang = list(map(float, ang_str.split('  ')))
        self.assertTrue(np.allclose(ang, lengths_ang[3:]))

        num_str = page.num_atoms
        num_str = num_str.replace('No. atoms ', '')

        self.assertEqual(int(num_str), 27)

    def test_to_from_dict(self):
        page = MCMainPage()

        num_fields = 0
        for item in page.walk():
            if isinstance(item, TextInput):
                num_fields += 1

        page.ids.eciFileInput.text = 'Some file'
        page.ids.sizeInput.text = '5'
        dct = page.to_dict()
        self.assertEqual(num_fields, len(dct))
        page2 = MCMainPage()
        page2.from_dict(dct)

        self.assertEqual(page2.ids.eciFileInput.text,
                         page.ids.eciFileInput.text)
        self.assertEqual(page2.ids.sizeInput.text,
                         page.ids.sizeInput.text)


if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("clease_gui.kv")
    unittest.main()
