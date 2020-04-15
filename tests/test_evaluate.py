from clease import (CEBulk, NewStructures, Evaluate,
                    Concentration)
from ase.calculators.emt import EMT
from ase.db import connect
import numpy as np
import unittest
from clease.tools import update_db


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

    def test_filter_cname_on_size(self):
        input = {"cf_name": ['c1_d0001_0', 'c1_d0002_0', 'c2_d0001_0',
                             'c2_d0002_0', 'c4_d0004_0', 'c3_d0005_0',
                             'c7_d0006_0'],
                 "dist": [0, 0, 1, 2, 4, 5, 6]}
        true_list = []
        predict_dict = {"max_cluster": [2, 3, 4, 1],
                        "result": [['c1_d0001_0', 'c1_d0002_0', 'c2_d0001_0',
                                    'c2_d0002_0'],
                                   ['c1_d0001_0', 'c1_d0002_0', 'c2_d0001_0',
                                    'c2_d0002_0', 'c3_d0005_0'],
                                   ['c1_d0001_0', 'c1_d0002_0', 'c2_d0001_0',
                                    'c2_d0002_0', 'c4_d0004_0', 'c3_d0005_0'],
                                   ['c1_d0001_0', 'c1_d0002_0']]}

        evaluator = Evaluate(settings=bc_setting)
        evaluator.cf_names = input['cf_name']
        for max_number in predict_dict['max_cluster']:
            true_list.append(evaluator._filter_cname_on_size(max_number))

        for true, predict in zip(true_list, predict_dict['result']):
            self.assertListEqual(true, predict)

    def test_distance_from_name(self):
        input = {"cf_name": ['c1_d0001_0', 'c1_d0002_0', 'c2_d0001_0',
                             'c2_d0002_0', 'c4_d0004_0', 'c3_d0005_0',
                             'c7_d0006_0'],
                 "dist": [0, 0, 1, 2, 4, 5, 6]}
        evaluator = Evaluate(settings=bc_setting)
        evaluator.cf_names = input['cf_name']
        predict_list = evaluator._distance_from_names()
        self.assertListEqual(input['dist'], predict_list)

    def test_cv(self):
        input_type = {'input_matrix': np.array([[1, 1, 1, 1, 1],
                                                [1, 2, 2, 2, 2],
                                                [1, 3, 3, 3, 3]]),
                      'result_matrix': np.array([1, 2, 3]),
                      'true_list': np.array([1, 2, 3])}

        # loocv
        evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
        evaluator.cf_matrix = input_type['input_matrix']
        evaluator.e_dft = input_type['result_matrix']
        evaluator.weight_matrix = np.eye(len(input_type['result_matrix']))
        loocv_result = evaluator.loocv()
        self.assertAlmostEqual(round(loocv_result, 6), 0)
        for true, predict in zip(input_type['true_list'],
                                 evaluator.e_pred_loo):
            self.assertAlmostEqual(true, round(predict, 5))

        # loocv_fast
        evaluator = Evaluate(bc_setting,
                             fitting_scheme="l2",
                             alpha=1E-6,
                             scoring_scheme="loocv_fast")
        evaluator.cf_matrix = input_type['input_matrix']
        evaluator.e_dft = input_type['result_matrix']
        evaluator.weight_matrix = np.eye(len(input_type['result_matrix']))
        fast_loocv_result = evaluator.loocv_fast()
        self.assertAlmostEqual(round(fast_loocv_result, 6), 0)
        for true, predict in zip(input_type['true_list'],
                                 evaluator.e_pred_loo):
            self.assertAlmostEqual(true, round(predict, 5))

        # k-fold cv
        evaluator = Evaluate(bc_setting,
                             fitting_scheme="l2",
                             alpha=1E-6,
                             scoring_scheme="k-fold")
        evaluator.nsplits = 3
        evaluator.cf_matrix = input_type['input_matrix']
        evaluator.e_dft = input_type['result_matrix']
        evaluator.weight_matrix = np.eye(len(input_type['result_matrix']))
        kfold_result = evaluator.k_fold_cv()
        self.assertAlmostEqual(round(kfold_result, 6), 0)

        # get_cv
        predict_input = ['loocv', 'loocv_fast', 'k-fold']
        true_list = []
        evaluator.nsplits = 3
        for predict_cv in predict_input:
            evaluator.scoring_scheme = predict_cv
            true_list.append(evaluator.get_cv_score())

        self.assertEqual(loocv_result*1000, true_list[0])
        self.assertEqual(fast_loocv_result*1000, true_list[1])
        self.assertEqual(kfold_result*1000, true_list[2])

    def test_error(self):
        input_type = {'input_matrix': np.array([[1, 1, 1, 1, 1],
                                                [1, 2, 2, 2, 2],
                                                [1, 3, 3, 3, 3]]),
                      'result_matrix': np.array([1, 2, 3]),
                      'true_list': np.array([1, 2, 3])}
        evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
        evaluator.cf_matrix = input_type['input_matrix']
        evaluator.e_dft = input_type['result_matrix']
        evaluator.weight_matrix = np.eye(len(input_type['result_matrix']))

        rmse_error = evaluator.rmse()
        self.assertAlmostEqual(rmse_error, 0)

        mae_error = evaluator.mae()
        self.assertAlmostEqual(mae_error, 0)

    def test_alpha_cv(self):
        predict_list = []
        true_list = []
        input_type = {'min_alpha': np.array([1, 2, 3, 4, 5, 6]),
                      'max_alpha': np.array([2, 3, 4, 5, 6, 7]),
                      'scheme': ['loocv', 'loocv_fast',
                                 'loocv', 'loocv', 'loocv_fast']}

        for scheme, min, max in zip(input_type['scheme'],
                                    input_type['min_alpha'],
                                    input_type['max_alpha']):
            evaluator = Evaluate(bc_setting, scoring_scheme=scheme)
            [alpha, cv] = evaluator.alpha_CV(alpha_min=min, alpha_max=max)
            alpha_min = alpha[np.argmin(cv)]
            predict_list.append(np.min(cv))
            evaluator = Evaluate(bc_setting,
                                 scoring_scheme=scheme, alpha=alpha_min)
            if scheme == 'loocv':
                true_list.append(evaluator.loocv())
            elif scheme == 'loocv_fast':
                true_list.append(evaluator.loocv_fast())
        for true, predict in zip(true_list, predict_list):
            self.assertAlmostEqual(true, predict)

    def test_cname_circum_dia(self):

        db_name = "test_aucu_evaluate1.db"
        basis_elements = [['Au', 'Cu']]
        conc = Concentration(basis_elements=basis_elements)
        setting = CEBulk(concentration=conc,
                         crystalstructure='fcc',
                         a=4.05,
                         size=[3, 3, 3],
                         db_name=db_name,
                         max_cluster_dia=[6.0, 6.0, 5.0])

        newstruct = NewStructures(setting, struct_per_gen=3)
        newstruct.generate_initial_pool()
        calc = EMT()
        database = connect(db_name)

        for row in database.select([("converged", "=", False)]):
            atoms = row.toatoms()
            atoms.set_calculator(calc)
            atoms.get_potential_energy()
            update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)

        input = {"cf_name": ['c0', 'c1',
                             'c2_d0001_0_0', 'c3_d0002_0_0', 'c4_d0001_0_0',
                             'c4_d0002_0_0', 'c4_d0004_0_0'],
                 "true": ['c0', 'c1', 'c2_d0001_0_0',
                          'c3_d0002_0_0', 'c4_d0001_0_0', 'c4_d0002_0_0']}

        evaluator = Evaluate(settings=setting)
        evaluator.cf_names = input['cf_name']

        true_list = evaluator._filter_cname_circum_dia(setting.max_cluster_dia)
        self.assertListEqual(input['true'], true_list)

    def test_cv_for_alpha(self):
        """
        Temporary name of new 'alpha_cv' method.
        If the document is updated related to evaluate class,
        The method name should be changed.
        """
        evaluator = Evaluate(settings=bc_setting, fitting_scheme='l1')
        alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        true_list = []
        evaluator.cv_for_alpha(alphas)
        for alpha in alphas:
            evaluator.scheme.alpha = alpha
            true_list.append(evaluator.loocv())
        predict_list = evaluator.cv_scores
        predict_list = [tmp_i['cv'] for tmp_i in predict_list]
        self.assertListEqual(true_list, predict_list)

    def test_get_energy_predict(self):
        evaluator = Evaluate(settings=bc_setting, fitting_scheme='l1')
        evaluator.get_eci()
        true_list = evaluator.cf_matrix.dot(evaluator.eci)
        predict_list = evaluator.get_energy_predict()
        self.assertListEqual(true_list.tolist(), predict_list.tolist())

    def test_get_energy_predict(self):
        evaluator = Evaluate(settings=bc_setting, fitting_scheme='l1')
        evaluator.get_eci()
        true_list = evaluator.cf_matrix.dot(evaluator.eci)
        predict_list = evaluator.get_energy_predict()
        self.assertListEqual(true_list.tolist(), predict_list.tolist())

    def test_subtract_predict_dft(self):
        evaluator = Evaluate(settings=bc_setting, fitting_scheme='l1')
        evaluator.get_eci()
        e_pred = evaluator.get_energy_predict()
        true_delta = evaluator.e_dft - e_pred
        predict_delta = evaluator.subtract_predict_dft()
        self.assertListEqual(true_delta.tolist(), predict_delta.tolist())

    def test_subtract_predict_dft_loo(self):
        evaluator = Evaluate(settings=bc_setting, fitting_scheme='l1')
        evaluator.loocv()
        true_delta = evaluator.e_dft - evaluator.e_pred_loo
        predict_delta = evaluator.subtract_predict_dft_loo()
        self.assertListEqual(true_delta.tolist(), predict_delta.tolist())

    def test_get_eci_by_size(self):
        evaluator = Evaluate(settings=bc_setting, fitting_scheme='l1')
        evaluator.get_eci()
        name_list = []
        distance_list = []
        eci_list = []
        distances = evaluator._distance_from_names()

        # Structure the ECIs in terms by size

        for name, distance, eci in zip(evaluator.cf_names, distances, evaluator.eci):
            distance_list.append(distance)
            eci_list.append(eci)
            name_list.append(name)
        dict_eval = evaluator.get_eci_by_size()

        predict_name = []
        predict_eci = []
        predict_distance = []

        for tmpi in dict_eval:
            predict_name = predict_name + dict_eval[tmpi]['name']
            predict_eci = predict_eci + dict_eval[tmpi]['eci']
            predict_distance = predict_distance + dict_eval[tmpi]['distance']

        self.assertEqual(name_list, predict_name)
        self.assertEqual(eci_list, predict_eci)
        self.assertEqual(distance_list, predict_distance)

if __name__ == '__main__':
    unittest.main()
