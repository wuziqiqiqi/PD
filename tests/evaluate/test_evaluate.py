import os
import json
import pytest
import numpy as np
from clease import Evaluate


@pytest.fixture
def make_eval(bc_setting):

    def _make_eval(**kwargs):
        evaluator = Evaluate(settings=bc_setting, **kwargs)
        return evaluator

    return _make_eval


def test_filter_cname_on_size(make_eval):
    inputs = {
        "cf_name": [
            'c1_d0001_0', 'c1_d0002_0', 'c2_d0001_0', 'c2_d0002_0', 'c4_d0004_0', 'c3_d0005_0',
            'c7_d0006_0'
        ],
        "dist": [0, 0, 1, 2, 4, 5, 6]
    }
    true_list = []
    predict_dict = {
        "max_cluster": [2, 3, 4, 1],
        "result":
            [['c1_d0001_0', 'c1_d0002_0', 'c2_d0001_0', 'c2_d0002_0'],
             ['c1_d0001_0', 'c1_d0002_0', 'c2_d0001_0', 'c2_d0002_0', 'c3_d0005_0'],
             ['c1_d0001_0', 'c1_d0002_0', 'c2_d0001_0', 'c2_d0002_0', 'c4_d0004_0', 'c3_d0005_0'],
             ['c1_d0001_0', 'c1_d0002_0']]
    }

    evaluator = make_eval()
    evaluator.cf_names = inputs['cf_name']
    for max_number in predict_dict['max_cluster']:
        true_list.append(evaluator._filter_cname_on_size(max_number))

    for true, predict in zip(true_list, predict_dict['result']):
        assert true == predict


def test_distance_from_name(make_eval):
    inputs = {
        "cf_name": [
            'c1_d0001_0', 'c1_d0002_0', 'c2_d0001_0', 'c2_d0002_0', 'c4_d0004_0', 'c3_d0005_0',
            'c7_d0006_0'
        ],
        "dist": [0, 0, 1, 2, 4, 5, 6]
    }
    evaluator = make_eval()
    evaluator.cf_names = inputs['cf_name']
    predict_list = evaluator._distance_from_names()
    assert inputs['dist'] == predict_list


def test_cv(make_eval):
    input_type = {
        'input_matrix': np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 2], [1, 3, 3, 3, 3]]),
        'result_matrix': np.array([1, 2, 3]),
        'true_list': np.array([1, 2, 3])
    }

    # Wrapper helper function for initializing an evaluator
    def make_evaluator(fitting_scheme="l2", alpha=1E-6, **kwargs):
        evaluator = make_eval(fitting_scheme=fitting_scheme, alpha=alpha, **kwargs)
        evaluator.cf_matrix = input_type['input_matrix']
        evaluator.e_dft = input_type['result_matrix']
        evaluator.weight_matrix = np.eye(len(input_type['result_matrix']))
        return evaluator

    # loocv
    evaluator = make_evaluator()
    loocv_result = evaluator.loocv()
    assert round(loocv_result, 6) == pytest.approx(0.0)
    for true, predict in zip(input_type['true_list'], evaluator.e_pred_loo):
        assert true == pytest.approx(round(predict, 5))

    # loocv_fast
    evaluator = make_evaluator(scoring_scheme="loocv_fast")
    fast_loocv_result = evaluator.loocv_fast()
    assert round(fast_loocv_result, 6) == pytest.approx(0.0)
    for true, predict in zip(input_type['true_list'], evaluator.e_pred_loo):
        assert true == pytest.approx(round(predict, 5))

    # k-fold cv
    evaluator = make_evaluator(scoring_scheme="k-fold")
    evaluator.nsplits = 3
    kfold_result = evaluator.k_fold_cv()
    assert kfold_result == pytest.approx(0.0, abs=1e-6)

    # get_cv
    predict_input = ['loocv', 'loocv_fast', 'k-fold']
    true_list = []
    evaluator.nsplits = 3
    for predict_cv in predict_input:
        evaluator.scoring_scheme = predict_cv
        true_list.append(evaluator.get_cv_score())

    assert loocv_result * 1000 == true_list[0]
    assert fast_loocv_result * 1000 == true_list[1]
    assert kfold_result * 1000 == true_list[2]


def test_error(make_eval):
    input_type = {
        'input_matrix': np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 2], [1, 3, 3, 3, 3]]),
        'result_matrix': np.array([1, 2, 3]),
        'true_list': np.array([1, 2, 3])
    }
    evaluator = make_eval(fitting_scheme="l2", alpha=1E-6)
    evaluator.cf_matrix = input_type['input_matrix']
    evaluator.e_dft = input_type['result_matrix']
    evaluator.weight_matrix = np.eye(len(input_type['result_matrix']))

    rmse_error = evaluator.rmse()
    assert rmse_error == pytest.approx(0.0, abs=1e-8)

    mae_error = evaluator.mae()
    assert mae_error == pytest.approx(0.0, abs=1e-8)


def test_alpha_cv(make_eval):
    predict_list = []
    true_list = []
    input_type = {
        'min_alpha': np.array([1, 2, 3, 4, 5, 6]),
        'max_alpha': np.array([2, 3, 4, 5, 6, 7]),
        'scheme': ['loocv', 'loocv_fast', 'loocv', 'loocv', 'loocv_fast']
    }

    for scheme, min_alph, max_alph in zip(input_type['scheme'], input_type['min_alpha'],
                                          input_type['max_alpha']):
        evaluator = make_eval(scoring_scheme=scheme)
        [alpha, cv] = evaluator.alpha_CV(alpha_min=min_alph, alpha_max=max_alph)
        alpha_min = alpha[np.argmin(cv)]
        predict_list.append(np.min(cv))
        evaluator = make_eval(scoring_scheme=scheme, alpha=alpha_min)

        res = getattr(evaluator, scheme)()
        true_list.append(res)
    for true, predict in zip(true_list, predict_list):
        assert true == pytest.approx(predict)


def test_cname_circum_dia(make_eval):
    inputs = {
        "cf_name": [
            'c0', 'c1', 'c2_d0001_0_0', 'c3_d0002_0_0', 'c4_d0001_0_0', 'c4_d0002_0_0',
            'c4_d0004_0_0'
        ],
        "true": ['c0', 'c1', 'c2_d0001_0_0', 'c4_d0001_0_0', 'c4_d0002_0_0']
    }

    evaluator = make_eval()
    settings = evaluator.settings
    evaluator.cf_names = inputs['cf_name']

    true_list = evaluator._filter_cname_circum_dia(settings.max_cluster_dia)

    assert inputs['true'] == true_list


def test_cv_for_alpha(make_eval):
    """
    Temporary name of new 'alpha_cv' method.
    If the document is updated related to evaluate class,
    The method name should be changed.
    """
    evaluator = make_eval(fitting_scheme='l1')
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    true_list = []
    evaluator.cv_for_alpha(alphas)
    for alpha in alphas:
        evaluator.scheme.alpha = alpha
        true_list.append(evaluator.loocv())
    predict_list = evaluator.cv_scores
    predict_list = [tmp_i['cv'] for tmp_i in predict_list]
    assert true_list == predict_list


def test_get_energy_predict(make_eval):
    evaluator = make_eval(fitting_scheme='l1')
    evaluator.get_eci()
    true_list = evaluator.cf_matrix.dot(evaluator.eci)
    predict_list = evaluator.get_energy_predict()
    assert true_list.tolist() == predict_list.tolist()


def test_save_eci(make_eval, make_tempfile):
    evaluator = make_eval()

    # Save with extension
    fname = make_tempfile('eci.json')
    evaluator.save_eci(fname=fname)
    assert os.path.isfile(fname)

    # Save without extension
    fname = make_tempfile('eci_no_ext')
    evaluator.save_eci(fname=fname)
    assert os.path.isfile(str(fname) + '.json')
    # Extensionless file should not exist in any form
    assert not os.path.exists(fname)


def test_load_eci(make_eval, make_tempfile):
    evaluator = make_eval()

    eci = evaluator.get_eci_dict()

    assert len(eci) > 0

    fname = make_tempfile('eci.json')
    evaluator.save_eci(fname=fname)

    # Load the ECI dict
    with open(fname) as file:
        eci_loaded = json.load(file)

    assert eci.keys() == eci_loaded.keys()

    for k, v in eci.items():
        assert pytest.approx(v) == eci_loaded[k]


def test_subtract_predict_dft(make_eval):
    evaluator = make_eval(fitting_scheme='l1')
    evaluator.get_eci()
    e_pred = evaluator.get_energy_predict()
    true_delta = evaluator.e_dft - e_pred
    predict_delta = evaluator.subtract_predict_dft()
    assert true_delta.tolist() == predict_delta.tolist()


def test_subtract_predict_dft_loo(make_eval):
    evaluator = make_eval(fitting_scheme='l1')
    evaluator.loocv()
    true_delta = evaluator.e_dft - evaluator.e_pred_loo
    predict_delta = evaluator.subtract_predict_dft_loo()
    assert true_delta.tolist() == predict_delta.tolist()


def test_get_eci_by_size(make_eval):
    evaluator = make_eval(fitting_scheme='l1')
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
    assert len(dict_eval) > 0

    predict_name = []
    predict_eci = []
    predict_distance = []

    for tmpi in dict_eval:
        predict_name = predict_name + dict_eval[tmpi]['name']
        predict_eci = predict_eci + dict_eval[tmpi]['eci']
        predict_distance = predict_distance + dict_eval[tmpi]['distance']

    assert name_list == predict_name
    assert eci_list == predict_eci
    assert distance_list == predict_distance
