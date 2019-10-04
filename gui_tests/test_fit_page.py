import unittest
from unittest.mock import patch, MagicMock

from clease.gui.fitPage import FitPage
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
import numpy as np
import time


class FitPageTests(unittest.TestCase):
    def eci_popup(self, app):
        screen = app.root.ids.sm.get_screen('Fit')

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadEciInput.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertEqual(screen._pop_up.title, "Load ECI file")
        screen.dismiss_popup()

    def open_fit_alg_editors(self, app):
        screen = app.root.ids.sm.get_screen('Fit')
        spinner = screen.ids.fitAlgSpinner

        last_pop_title = ""
        for value in spinner.values:
            spinner.text = value
            self.assertTrue(screen._pop_up is None)
            screen.ids.fitEditorButton.dispatch('on_release')
            self.assertFalse(screen._pop_up is None)
            new_title = screen._pop_up.title

            # We don't explicitly check the title, but in order to make
            # sure that the pop up actually changes we check that the
            # new title is different from the previous
            self.assertNotEqual(new_title, last_pop_title)
            screen.dismiss_popup()
            last_pop_title = new_title

    @patch('clease.gui.fitPage.Evaluate')
    @patch('clease.gui.fitPage.GAFit')
    @patch('clease.gui.fitPage.App')
    def test_ga_fit(self, app_mock, ga_fit_eval_mock, eval_mock):
        # Specify required return values
        eval_mock.return_value.eci = np.array([1.0, 2.0])
        eval_mock.return_value.get_eci_dict.return_value = {'c0': 1.0,
                                                            'c2_0': 2.0}
        eval_mock.return_value.cf_matrix = np.array([[1.0, 1.0], [0.2, -0.1]])
        eval_mock.return_value.e_dft = np.array([-0.5, -0.8])
        eval_mock.return_value.e_pred_loo = np.array([-0.1, 0.9])
        eval_mock.return_value.get_cv_score.return_value = 0.2
        eval_mock.return_value.rmse.return_value = 0.1
        eval_mock.return_value.mae.return_value = 0.08

        app_mock.get_running_app = MagicMock(root=MagicMock(
            ids=MagicMock(text='')))

        #exit()
        # Set fitting algorithm to GA
        page = FitPage()

        # Trigger the on_enter command
        page.on_enter()

        # Change algorithm
        page.ids.fitAlgSpinner.text = 'Genetic Algorithm'

        # Launch the fit alg editor
        page.ids.fitEditorButton.dispatch('on_release')

        # Close the fit alg editor
        page._pop_up.content.ids.closeButton.dispatch('on_release')

        # Confirm that there is no data in the plots
        self.assertEqual(page.energy_plot.points, [])
        self.assertTrue(all(plot.points, []) for plot in page.eci_plots)

        # Launch a fitting, with a fake evaluate and GAFit object
        page.ids.fitButton.dispatch('on_release')

        # Add a 1/100 sec latency
        time.sleep(0.01)

        # Confirm that plots are populated
        expect = [(-0.5, 3500.0), (-0.8, 800.0)]
        self.assertEqual(len(expect), len(page.energy_plot.points))
        tol = 1E-6
        self.assertTrue(
            all(map(lambda x:
                abs(x[0][0] - x[1][0]) < 1E-6 and abs(x[0][1] - x[1][1]) < tol,
                zip(expect, page.energy_plot.points))))

        # Check that the ECI plot has been updated
        self.assertEqual(len(page.eci_plots[0].points), 1)
        self.assertAlmostEqual(page.eci_plots[0].points[0][1], 2.0)

    def run_with_app(self, app):
        self.eci_popup(app)
        self.open_fit_alg_editors(app)


if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("fitLayout.kv")
    Builder.load_file("gaEditor.kv")
    unittest.main()
