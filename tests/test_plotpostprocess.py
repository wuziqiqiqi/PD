from clease import (CEBulk, NewStructures, Evaluate,
                    Concentration)
from ase.calculators.emt import EMT
from ase.db import connect
import numpy as np
import unittest
from clease.tools import update_db
from clease.plot_post_process import plot_fit, plot_fit_residual, plot_eci


def create_database():
    db_name = "test_aucu_evaluate.db"
    basis_elements = [['Au', 'Cu']]
    conc = Concentration(basis_elements=basis_elements)
    bc_setting = CEBulk(concentration=conc, crystalstructure='fcc',
                        a=4.05, size=[3, 3, 3], db_name=db_name)
    newstruct = NewStructures(bc_setting, struct_per_gen=3)
    newstruct.generate_initial_pool()
    calc = EMT()
    database = connect(db_name)

    for row in database.select([("converged", "=", False)]):
        atoms = row.toatoms()
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)

    return bc_setting


bc_setting = create_database()


class TestEvaluate(unittest.TestCase):

    def test_plot_fit(self):
        predict = {"title": "plot_FIT_TEST",
                   "xlabel": "E_DFT (eV/atom)",
                   "ylabel": "DFT_FIT"}
        plot_args = {"title": predict["title"],
                     "ylabel": predict["ylabel"]}
        evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
        evaluator.get_eci()

        # x-axis values calculated by get_energy_predict
        predict['x_axis_value'] = \
            np.ndarray.tolist(evaluator.get_energy_predict())
        # y-axis values calculated by get_energy_predict
        predict['y_axis_value'] = \
            np.ndarray.tolist(evaluator.e_dft)

        fig = plot_fit(evaluator, plot_args)
        self.assertTrue('LOOCV' in fig.get_axes()[0].texts[0].get_text())
        self.assertEqual(predict["title"], fig.get_axes()[0].get_title())
        self.assertEqual(predict["xlabel"], fig.get_axes()[0].get_xlabel())
        self.assertEqual(predict["ylabel"], fig.get_axes()[0].get_ylabel())
        self.assertListEqual(predict['x_axis_value'],
                             np.ndarray.tolist(fig.gca().lines[1].get_xdata()))
        self.assertListEqual(predict['y_axis_value'],
                             np.ndarray.tolist(fig.gca().lines[1].get_ydata()))

    def test_plot_fit_residual(self):
        predict = {"title": "plot_FIT_TEST",
                   "xlabel": "#OCC",
                   "ylabel": "DFT_FIT"}
        plot_args = {"title": predict["title"],
                     "ylabel": predict["ylabel"]}
        evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
        evaluator.get_eci()

        # y-axis values calculated by subtract_predict_dft
        predict['delta_e'] = np.ndarray.tolist(evaluator.subtract_predict_dft())

        fig = plot_fit_residual(evaluator, plot_args)
        self.assertEqual(predict["title"], fig.get_axes()[0].get_title())
        self.assertEqual(predict["xlabel"], fig.get_axes()[1].get_xlabel())
        self.assertEqual(predict["ylabel"], fig.get_axes()[0].get_ylabel())
        self.assertListEqual(predict['delta_e'],
                             np.ndarray.tolist(fig.get_children()[1]
                                               .get_lines()[1].get_ydata()))

    def test_plot_eci(self):
        predict = {"title": "plot_FIT_TEST",
                   "xlabel": "Cluster diameter ($n^{th}$ nearest neighbor)",
                   "ylabel": "DFT_FIT"}
        plot_args = {"title": predict["title"],
                     "ylabel": predict["ylabel"]}
        evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
        evaluator.get_eci()

        # x, y-axis values calculated by get_eci_by_size
        test_list = evaluator.get_eci_by_size()
        test_list = list(test_list.values())

        # x, y-axis values of 4 body cluster eci
        predict['eci'] = test_list[4]['eci']

        # x, y-axis values of axhline
        predict['axhline_xy'] = [[0.0, 0.0], [1.0, 0.0]]

        fig = plot_eci(evaluator, plot_args)
        self.assertEqual(predict["title"], fig.get_axes()[0].get_title())
        self.assertEqual(predict["xlabel"], fig.get_axes()[0].get_xlabel())
        self.assertEqual(predict["ylabel"], fig.get_axes()[0].get_ylabel())
        self.assertListEqual(predict['eci'],
                             np.ndarray.tolist(fig.gca().lines[5].get_ydata()))
        self.assertListEqual(predict['axhline_xy'],
                             np.ndarray.tolist(fig.gca().axhline().get_xydata()))


if __name__ == '__main__':
    unittest.main()
