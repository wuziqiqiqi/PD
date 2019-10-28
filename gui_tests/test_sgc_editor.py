import unittest
from clease.gui.sgcEditor import SGCEditor
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
import os


class TestSGCEditor(unittest.TestCase):
    def test_symbol_parsing(self):
        tests = [
            {
                'input': 'Au, Cu',
                'expect': ['Au', 'Cu']
            },
            {
                'input': 'Au,Cu',
                'expect': ['Au', 'Cu']
            },
            {
                'input': '[Au, Cu]',
                'expect': ['Au', 'Cu']
            },
            {
                'input': '(Au, Cu)',
                'expect': ['Au', 'Cu']
            },
            {
                'input': 'Al, Mg, X, Si',
                'expect': ['Al', 'Mg', 'X', 'Si']
            }
        ]
        editor = SGCEditor(symbols='Au, Cu', chem_pot='Au: 0.0')

        for i, test in enumerate(tests):
            editor.ids.symbolInput.text = test['input']
            value = editor.parse_symbols()
            msg = 'symbols: Failing for test {} '.format(i)
            msg += 'with input {} '.format(test['input'])
            msg += 'got {}, expected {}'.format(value, test['expect'])
            self.assertEqual(value, test['expect'], msg=msg)

    def test_chem_pot_parsing(self):
        tests = [
            {
                'input': 'c1_0: 0.1',
                'expect': {'c1_0': 0.1}
            },
            {
                'input': '{{c1_0: 0.1}}',
                'expect': {'c1_0': 0.1}
            },
            {
                'input': 'c1_0: 0.4, c1_2: 0.7',
                'expect': {'c1_0': 0.4, 'c1_2': 0.7}
            },
            {
                'input': 'c1_0: 0.4,c1_2: 0.7',
                'expect': {'c1_0': 0.4, 'c1_2': 0.7}
            }
        ]

        editor = SGCEditor(symbols='Au, Cu', chem_pot='Au: 0.0')
        for i, test in enumerate(tests):
            editor.ids.chemPotInput.text = test['input']
            chem_pot = editor.parse_chem_pot()
            msg = 'Failed for test {} '.format(i)
            msg += 'Input {}, output {}'.format(test['input'], chem_pot)
            msg += 'Expected: {}'.format(test['expect'])
            self.assertDictEqual(chem_pot, test['expect'])

    def test_load_values(self):
        editor = SGCEditor(symbols='Au, Cu', chem_pot='Au: 0.0')
        editor.ids.symbolInput.text = 'Some random symbols'
        editor.ids.chemPotInput.text = 'Chemical potentials go here!'
        editor.backup()

        # Open a new editor and confirm that the texr mathes
        editor2 = SGCEditor(symbols='Au, Cu', chem_pot='Au: 0.0')
        self.assertEqual(editor.ids.symbolInput.text,
                         editor2.ids.symbolInput.text)
        self.assertEqual(editor.ids.chemPotInput.text,
                         editor2.ids.chemPotInput.text)
        os.remove(editor.fname)


if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("cleaseGUILayout.kv")
    unittest.main()
