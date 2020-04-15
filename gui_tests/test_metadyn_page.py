import unittest
from unittest.mock import patch, MagicMock
from clease.gui.meta_dyn_page import MetaDynPage
from clease.gui.help_message_popup import HelpMessagePopup
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
from clease.gui.constants import META_DYN_MSG
import json
import os
from ase.build import bulk
from ase.calculators.interface import Calculator
import numpy as np


class SimpleCalc(Calculator):
    last_change = None
    atoms = None

    def get_energy_given_change(self, change):
        self.last_change = change
        for c in change:
            self.atoms[c[0]].symbol = c[2]
        return 0.0

    def calculate(self, *args):
        return 0.0

    def clear_history(self):
        pass

    def get_singlets(self):
        return np.array([])

    @property
    def results(self):
        return {'energy': 0.0}

    def restore(self):
        pass


def fake_attach(**kwargs):
    calc = SimpleCalc()
    calc.atoms = kwargs['atoms']
    kwargs['atoms'].set_calculator(calc)
    return kwargs['atoms']


class TestMetaDynPage(unittest.TestCase):
    @patch('clease.gui.meta_dyn_page.App')
    def test_help_messages(self, app_mock):
        page = MetaDynPage()

        counter = 0
        for child in page.ids.column2.children:
            if not hasattr(child, 'text'):
                continue
            if child.text == '?':
                child.dispatch('on_release')
                msg = 'Popup not set for button {}'.format(counter)
                msg += 'Popup is: {}'.format(page._pop_up)
                self.assertIsInstance(page._pop_up.content, HelpMessagePopup,
                                      msg=msg)
                page.dismiss_popup()
                msg = 'Popup not dismissed for button {}'.format(counter)
                self.assertIsNone(page._pop_up)
                counter += 1

    @patch('clease.gui.meta_dyn_page.App')
    def test_to_from_dict(self, app_mock):
        page = MetaDynPage()
        page.ids.tempInput.text = '800'
        page.ids.varSpinner.text = 'SomeVariable'
        page.ids.varMinInput.text = '0.5'
        page.ids.varMaxInput.text = '0.8'
        page.ids.flatInput.text = '0.9'
        page.ids.nbinsInput.text = '60'
        page.ids.backupInput.text = 'metadyn'
        page.ids.ensembleSpinner.text = 'some random ensemble'
        page.ids.modInput.text = '0.02'
        page.ids.plotIntInput.text = '20'

        dct = page.to_dict()
        page2 = MetaDynPage()
        page2.from_dict(dct)
        self.assertEqual(page.ids.tempInput.text, page2.ids.tempInput.text)
        self.assertEqual(page.ids.varSpinner.text, page2.ids.varSpinner.text)
        self.assertEqual(page.ids.varMinInput.text, page2.ids.varMinInput.text)
        self.assertEqual(page.ids.varMaxInput.text, page2.ids.varMaxInput.text)
        self.assertEqual(page.ids.flatInput.text, page2.ids.flatInput.text)
        self.assertEqual(page.ids.nbinsInput.text, page2.ids.nbinsInput.text)
        self.assertEqual(page.ids.backupInput.text, page2.ids.backupInput.text)
        self.assertEqual(page.ids.ensembleSpinner.text,
                         page2.ids.ensembleSpinner.text)
        self.assertEqual(page.ids.modInput.text, page2.ids.modInput.text)
        self.assertEqual(page.ids.plotIntInput.text,
                         page2.ids.plotIntInput.text)

    @patch('clease.gui.meta_dyn_page.App')
    def test_observer_popups(self, app_mock):
        page = MetaDynPage()
        page.unique_symbols = lambda: ['Au', 'Cu']

        expectations = {
            'Concentration': {
                'name': 'Concentration',
                'element': 'Au'
            }
        }

        for name in page.ids.varSpinner.values:
            page.ids.varSpinner.text = name
            page.ids.launchObsEditorButton.dispatch('on_release')

            msg = 'Popup not opened for observer {}'.format(name)
            self.assertIsNotNone(page._pop_up, msg=msg)
            if name == 'Concentration':
                page._pop_up.content.ids.elementInput.text = 'Au'
            page._pop_up.content.ids.closeButton.dispatch('on_release')

            expect = expectations[name]
            self.assertDictEqual(expect, page.observer_params)

    @patch('clease.gui.meta_dyn_page.App')
    def test_ensemble_popups(self, _):
        page = MetaDynPage()
        page.unique_symbols = lambda: ['Au', 'Cu', 'X']
        expect = {
            'Semi-grand canonical': {
                'name': 'Semi-grand canonical',
                'symbols': ['Au', 'Cu', 'X'],
                'chem_pot': {'Au': 0.0, 'Cu': 0.0}
            }
        }

        for ensemble in page.ids.ensembleSpinner.values:
            page.ids.ensembleSpinner.text = ensemble
            page.ids.launchEnsembleEditor.dispatch('on_release')
            msg = 'Popup not set for ensemble {}'.format(ensemble)
            self.assertIsNotNone(page._pop_up, msg=msg)
            page._pop_up.content.ids.closeButton.dispatch('on_release')
            self.assertDictEqual(expect[ensemble], page.ensemble_params)
            msg = 'Popup not closed for ensemble {}'.format(ensemble)
            self.assertIsNone(page._pop_up, msg=msg)

    @patch('clease.gui.meta_dyn_runner.attach_calculator')
    @patch('clease.gui.meta_dyn_page.App')
    def test_run(self, app_mock, attach_mock):
        status = MagicMock(text='')
        eci_file_input = MagicMock(text='')
        size_input = MagicMock(text='4')
        other_ids = MagicMock(
            eciFileInput=eci_file_input, sizeInput=size_input)
        ext_screen = MagicMock(ids=other_ids)

        sm = MagicMock(get_screen=MagicMock(return_value=ext_screen))

        # Fake screen manager by re-using the old one
        ext_screen.ids.sm = sm

        ids = MagicMock(status=status, sm=sm)
        root = MagicMock(settings=None, ids=ids,
                         active_template_is_mc_cell=True)
        app_mock.get_running_app = MagicMock(return_value=MagicMock(root=root))
        attach_mock.side_effect = fake_attach

        page = MetaDynPage()
        page.unique_symbols = lambda: ['Au', 'Cu']
        page.ids.maxSweepsInput.text = '2'

        # Settings is None
        page.ids.runButton.dispatch('on_release')
        self.assertEqual(status.text, META_DYN_MSG['settings_is_none'])

        # More than one basis (this test can be removed in the future)
        conc = MagicMock(basis_elements=[['Au', 'Cu'], ['X', 'Cu']])
        root.settings = MagicMock(concentration=conc)

        page.ids.runButton.dispatch('on_release')
        self.assertEqual(status.text, META_DYN_MSG['more_than_one_basis'])

        # No ECI file
        conc = MagicMock(basis_elements=[['Au', 'Cu', 'X']])
        bfs = [{'Au': 1.0, 'Cu': -1.0, 'X': 0.0},
               {'Au': -1.0, 'Cu': 0.0, 'X': 1.0}]
        root.settings = MagicMock(concentration=conc,
                                  atoms=bulk('Al')*(2, 2, 2),
                                  basis_functions=bfs)
        page.ids.runButton.dispatch('on_release')
        self.assertTrue(status.text.startswith(META_DYN_MSG['no_eci']))

        # Write an ECI file
        eci = {'c0': 1.0, 'c1_1': 2.0}
        fname = 'example_eci.json'
        with open(fname, 'w') as out:
            json.dump(eci, out)
        eci_file_input.text = fname

        # MC cell is already active
        page.ids.runButton.dispatch('on_release')
        self.assertEqual(status.text, META_DYN_MSG['mc_cell_is_template'])

        root.active_template_is_mc_cell = False

        # Variable editor has not been launced
        page.ids.runButton.dispatch('on_release')
        self.assertEqual(status.text, META_DYN_MSG['var_editor_not_launched'])

        # Trigger editor
        page.launch_observer_editor()
        page._pop_up.content.ids.closeButton.dispatch('on_release')
        page.ids.runButton.dispatch('on_release')

        # Ensemble editor has not been launcued
        self.assertEqual(status.text, META_DYN_MSG['launch_ens_editor'])

        # Trigger ensemble editor
        page.launch_ensemble_editor()
        page._pop_up.content.ids.closeButton.dispatch('on_release')
        page.ids.runButton.dispatch('on_release')
        os.remove(fname)

    @patch('clease.gui.meta_dyn_page.App')
    def test_abort(self, app_mock):
        status = MagicMock(text='')
        ids = MagicMock(status=status)
        root = MagicMock(ids=ids)
        app_mock.get_running_app = MagicMock(return_value=MagicMock(root=root))
        page = MetaDynPage()
        page.bind_meta_dyn_sampler(MagicMock())
        page.ids.abortButton.dispatch('on_release')
        self.assertEqual(status.text, META_DYN_MSG['abort_mc'])

    @patch('clease.gui.meta_dyn_page.App')
    def test_clear(self, app_mock):
        page = MetaDynPage()
        fname = 'test_clearbutton'
        page.ids.backupInput.text = fname
        fname += '.json'

        with open(fname, 'w') as out:
            json.dump({'some_data': 2}, out)

        self.assertTrue(os.path.exists(fname))
        page.ids.clearButton.dispatch('on_release')
        self.assertFalse(os.path.exists(fname))

    @patch('clease.gui.meta_dyn_page.App')
    def test_init_symbols_sgc(self, app_mock):
        settings = MagicMock()
        settings.unique_element_without_background = lambda: ['Au', 'Cu', 'X']
        root = MagicMock(settings=settings)
        page = MetaDynPage()
        app_mock.get_running_app = MagicMock(return_value=MagicMock(root=root))

        elems = page.unique_symbols()
        self.assertEqual(elems, ['Au', 'Cu', 'X'])


if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("clease_gui.kv")
    unittest.main()
