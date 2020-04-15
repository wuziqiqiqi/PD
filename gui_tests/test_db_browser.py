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
        self.assertEqual(str_rep, expect)
        os.remove(db_name)


if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("clease_gui.kv")
    unittest.main()
