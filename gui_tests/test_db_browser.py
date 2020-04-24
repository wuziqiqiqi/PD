import unittest
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
from clease.gui.db_browser import DbBrowser
from ase.atoms import Atoms
import os
from ase.db import connect
from ase.calculators.emt import EMT
import re


class TestDBBrowser(unittest.TestCase):
    def test_db_updates(self):
        db_name = 'test_db_updates.db'

        db = connect(db_name)
        atoms = Atoms('CO')
        calc = EMT()
        atoms.set_calculator(calc)
        energy1 = atoms.get_potential_energy()

        db.write(atoms, struct_type='initial', gen=0, name='CO_1',
                 converged=True, size='1x2x3')
        atoms[1].symbol = 'H'
        energy2 = atoms.get_potential_energy()
        db.write(atoms, struct_type='final', gen=1, name='CH_1',
                 converged=False, size='3x2x1')

        browser = DbBrowser(db_name=db_name)
        str_rep = browser.ids.text_field.text
        str_rep = str_rep.replace(' ', '')
        header = '|id|formula|calculator|energy|name|size|'
        header += 'gen|struct_type|converged|\n'
        expect = '|1|CO|emt|{:.3f}|CO_1|1x2x3|0|initial|1|\n'.format(energy1)
        line2 = '|2|CH|emt|{:.3f}|CH_1|3x2x1|1|final|0|\n'.format(energy2)
        expect += line2
        expect = header + expect
        self.assertEqual(str_rep, expect)

        # Test a query
        browser.ids.queryInput.text = 'formula=CH'
        browser.ids.queryInput.dispatch('on_text_validate')
        expect = header + line2
        str_rep = browser.ids.text_field.text
        str_rep = str_rep.replace(' ', '')
        print(str_rep)
        self.assertEqual(str_rep, expect)

        # Check that we extract the correct IDs for some tougher queries
        for i in range(20):
            db.write(Atoms('H'), gen=i, objNo=f'obj{int(i/4)}',
                     group=i % 4)

        tests = [
            {
                'query': 'id>15,id<18',
                'expect': [16, 17]
            },
            {
                'query': 'gen<5',
                'expect': [1, 2, 3, 4, 5, 6, 7]
            },
            {
                'query': 'id>11, group=0',
                'expect': [15, 19]
            },
            {
                'query': 'id>=11, objNo=obj2',
                'expect': [11, 12, 13, 14]
            }
        ]

        for test in tests:
            browser.ids.queryInput.text = test['query']
            browser.ids.queryInput.dispatch('on_text_validate')

            # Extract ids from the text displayed in the browser
            idsShown = []
            prog = re.compile(r"\|([1-9]+)\|")
            for line in browser.ids.text_field.text.split('\n'):
                line = line.replace(' ', '')
                m = prog.match(line)
                if m is None:
                    continue
                idsShown.append(int(m.group(1)))
            self.assertEqual(idsShown, test['expect'])

        os.remove(db_name)

    def test_select_conditions(self):
        db_name = 'test_select_cond.db'
        db = connect('test_select_cond.db')  # Initialize the DB
        db.write(Atoms('H'))
        browser = DbBrowser(db_name=db_name)

        operators = ['=', '!=', '>', '<', '>=', '<=']
        tests = []

        # Add single variable from systems table
        for operator in operators:
            tests.append({
                'query': f'id{operator}2',
                'system': [{
                    'sql': f'id{operator}?',
                    'value': '2',
                }],
                'kvp': {}
            })

        # Add single variable from kvps
        for operator in operators:
            tests.append({
                'query': f'gen{operator}3',
                'system': [],
                'kvp': {
                    'gen': {
                        'sql': [f'value{operator}?'],
                        'values': ['3']
                    }
                }
            })

        tests.append({
            'query': 'gen>=3,gen<10,converged=1',
            'system': [],
            'kvp': {
                'gen': {
                    'sql': ['value>=?', 'value<?'],
                    'values': ['3', '10']
                },
                'converged': {
                    'sql': ['value=?'],
                    'values': ['1']
                }
            }})

        for test in tests:
            syst, kvp = browser._get_select_conditions(test['query'])
            for item1, item2 in zip(syst, test['system']):
                self.assertDictEqual(item1, item2)
            self.assertDictEqual(kvp, test['kvp'])


if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("clease_gui.kv")
    unittest.main()
