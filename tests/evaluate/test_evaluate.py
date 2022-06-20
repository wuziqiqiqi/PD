import os
import json
import random
import pytest
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.calculators.emt import EMT
from clease.settings import CEBulk, Concentration
from clease import Evaluate, supports_alpha_cv
from clease import NewStructures
from clease.tools import update_db
from clease import regression
import clease.plot_post_process as pp


@pytest.fixture
def make_eval(bc_setting):
    def _make_eval(**kwargs):
        evaluator = Evaluate(settings=bc_setting, **kwargs)
        return evaluator

    return _make_eval


@pytest.fixture
def make_eval_with_bkg(bkg_ref_settings):
    def _make_eval(**kwargs):
        evaluator1 = Evaluate(settings=bkg_ref_settings[0], **kwargs)
        evaluator2 = Evaluate(settings=bkg_ref_settings[1], **kwargs)
        return evaluator1, evaluator2

    return _make_eval


@pytest.fixture
def bkg_ref_settings(make_tempfile):
    """
    This fixture creates two equivalent settings, one has two sub-lattices with one of
    them being the background (occupied by a single specie). The other settings contains
    the same structures as the first one but only has one sub-lattice (fcc).

    They should generate identical correlation functions if the background is simply ignored.
    """
    db_name_bkg = make_tempfile("temp_db_bkg_double.db")
    basis_elements = [["Au", "Cu"], ["Ag"]]
    conc = Concentration(basis_elements=basis_elements)
    settings_bkg = CEBulk(
        concentration=conc,
        crystalstructure="rocksalt",
        a=4.05,
        size=[3, 3, 3],
        db_name=db_name_bkg,
    )
    newstruct = NewStructures(settings_bkg, struct_per_gen=3)
    newstruct.generate_initial_pool()
    calc = EMT()

    with connect(db_name_bkg) as database:
        for row in database.select([("converged", "=", False)]):
            atoms = row.toatoms()
            atoms.calc = calc
            atoms.get_potential_energy()
            update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name_bkg)

    db_name_no_bkg = make_tempfile("temp_db_bkg_single.db")
    basis_elements = [["Au", "Cu"]]
    conc = Concentration(basis_elements=basis_elements)
    settings_no_bkg = CEBulk(
        concentration=conc,
        crystalstructure="fcc",
        a=4.05,
        size=[3, 3, 3],
        db_name=db_name_no_bkg,
    )
    newstruct = NewStructures(settings_no_bkg, struct_per_gen=3)
    with connect(db_name_bkg) as database:
        for row in database.select(converged=True):
            init_atoms = row.toatoms()
            final_atoms = database.get(id=row.key_value_pairs["final_struct_id"]).toatoms()
            new_init_atoms = init_atoms[atoms.numbers != 47]
            new_final_atoms = final_atoms[atoms.numbers != 47]
            new_final_atoms.calc = SinglePointCalculator(
                new_final_atoms, energy=final_atoms.get_potential_energy()
            )
            newstruct.insert_structure(new_init_atoms, new_final_atoms)

    yield (settings_bkg, settings_no_bkg)


def test_filter_cname_on_size(make_eval):
    inputs = {
        "cf_name": [
            "c1_d0001_0",
            "c1_d0002_0",
            "c2_d0001_0",
            "c2_d0002_0",
            "c4_d0004_0",
            "c3_d0005_0",
            "c7_d0006_0",
        ],
        "dist": [0, 0, 1, 2, 4, 5, 6],
    }
    true_list = []
    predict_dict = {
        "max_cluster": [2, 3, 4, 1],
        "result": [
            ["c1_d0001_0", "c1_d0002_0", "c2_d0001_0", "c2_d0002_0"],
            ["c1_d0001_0", "c1_d0002_0", "c2_d0001_0", "c2_d0002_0", "c3_d0005_0"],
            [
                "c1_d0001_0",
                "c1_d0002_0",
                "c2_d0001_0",
                "c2_d0002_0",
                "c4_d0004_0",
                "c3_d0005_0",
            ],
            ["c1_d0001_0", "c1_d0002_0"],
        ],
    }

    evaluator = make_eval()
    evaluator.cf_names = inputs["cf_name"]
    for max_number in predict_dict["max_cluster"]:
        true_list.append(evaluator._filter_cname_on_size(max_number))

    for true, predict in zip(true_list, predict_dict["result"]):
        assert true == predict


def test_distance_from_name(make_eval):
    inputs = {
        "cf_name": [
            "c1_d0001_0",
            "c1_d0002_0",
            "c2_d0001_0",
            "c2_d0002_0",
            "c4_d0004_0",
            "c3_d0005_0",
            "c7_d0006_0",
        ],
        "dist": [0, 0, 1, 2, 4, 5, 6],
    }
    evaluator = make_eval()
    evaluator.cf_names = inputs["cf_name"]
    predict_list = evaluator._distance_from_names()
    assert inputs["dist"] == predict_list


def test_cv(make_eval):
    input_type = {
        "input_matrix": np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 2], [1, 3, 3, 3, 3]]),
        "result_matrix": np.array([1, 2, 3]),
        "true_list": np.array([1, 2, 3]),
    }

    # Wrapper helper function for initializing an evaluator
    def make_evaluator(fitting_scheme="l2", alpha=1e-6, **kwargs):
        evaluator = make_eval(fitting_scheme=fitting_scheme, alpha=alpha, **kwargs)
        evaluator.cf_matrix = input_type["input_matrix"]
        evaluator.e_dft = input_type["result_matrix"]
        evaluator.weight_matrix = np.eye(len(input_type["result_matrix"]))
        return evaluator

    # loocv
    evaluator = make_evaluator()
    loocv_result = evaluator.loocv()
    assert round(loocv_result, 6) == pytest.approx(0.0)
    for true, predict in zip(input_type["true_list"], evaluator.e_pred_loo):
        assert true == pytest.approx(round(predict, 5))

    # loocv_fast
    evaluator = make_evaluator(scoring_scheme="loocv_fast")
    fast_loocv_result = evaluator.loocv_fast()
    assert round(fast_loocv_result, 6) == pytest.approx(0.0)
    for true, predict in zip(input_type["true_list"], evaluator.e_pred_loo):
        assert true == pytest.approx(round(predict, 5))

    # k-fold cv
    evaluator = make_evaluator(scoring_scheme="k-fold")
    evaluator.nsplits = 3
    kfold_result = evaluator.k_fold_cv()
    assert kfold_result == pytest.approx(0.0, abs=1e-6)

    # get_cv
    predict_input = ["loocv", "loocv_fast", "k-fold"]
    true_list = []
    evaluator.nsplits = 3
    for predict_cv in predict_input:
        evaluator.scoring_scheme = predict_cv
        true_list.append(evaluator.get_cv_score())

    assert loocv_result == pytest.approx(true_list[0])
    assert fast_loocv_result == pytest.approx(true_list[1])
    assert kfold_result == pytest.approx(true_list[2])


def test_error(make_eval):
    input_type = {
        "input_matrix": np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 2], [1, 3, 3, 3, 3]]),
        "result_matrix": np.array([1, 2, 3]),
        "true_list": np.array([1, 2, 3]),
    }
    evaluator = make_eval(fitting_scheme="l2", alpha=1e-6)
    evaluator.cf_matrix = input_type["input_matrix"]
    evaluator.e_dft = input_type["result_matrix"]
    evaluator.weight_matrix = np.eye(len(input_type["result_matrix"]))

    rmse_error = evaluator.rmse()
    assert rmse_error == pytest.approx(0.0, abs=1e-8)

    mae_error = evaluator.mae()
    assert mae_error == pytest.approx(0.0, abs=1e-8)


def test_alpha_cv(make_eval):
    predict_list = []
    true_list = []
    input_type = {
        "min_alpha": np.array([1, 2, 3, 4, 5, 6]),
        "max_alpha": np.array([2, 3, 4, 5, 6, 7]),
        "scheme": ["loocv", "loocv_fast", "loocv", "loocv", "loocv_fast"],
    }

    for scheme, min_alph, max_alph in zip(
        input_type["scheme"], input_type["min_alpha"], input_type["max_alpha"]
    ):
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
    evaluator = make_eval()

    true_list = evaluator._filter_cname_circum_dia([4, 5, 4])

    assert true_list == [
        "c0",
        "c1_0",
        "c2_d0000_0_00",
        "c3_d0000_0_000",
        "c3_d0001_0_000",
        "c4_d0000_0_0000",
    ]


def test_cv_for_alpha(make_eval):
    """
    Temporary name of new 'alpha_cv' method.
    If the document is updated related to evaluate class,
    The method name should be changed.
    """
    evaluator = make_eval(fitting_scheme="l1")
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    true_list = []
    evaluator.cv_for_alpha(alphas)
    for alpha in alphas:
        evaluator.scheme.alpha = alpha
        true_list.append(evaluator.loocv())
    predict_list = evaluator.cv_scores
    predict_list = [tmp_i["cv"] for tmp_i in predict_list]
    assert true_list == predict_list


def test_get_energy_predict(make_eval):
    evaluator = make_eval(fitting_scheme="l1")
    evaluator.fit()
    evaluator.get_eci()
    true_list = evaluator.cf_matrix.dot(evaluator.eci)
    predict_list = evaluator.get_energy_predict()
    assert true_list.tolist() == predict_list.tolist()


def test_save_eci(make_eval, make_tempfile):
    evaluator = make_eval()
    evaluator.fit()

    # Save with extension
    fname = make_tempfile("eci.json")
    evaluator.save_eci(fname=fname)
    assert os.path.isfile(fname)

    # Save without extension
    fname = make_tempfile("eci_no_ext")
    evaluator.save_eci(fname=fname)
    assert os.path.isfile(str(fname) + ".json")
    # Extensionless file should not exist in any form
    assert not os.path.exists(fname)


def test_load_eci(make_eval, make_tempfile):
    evaluator = make_eval()
    evaluator.fit()

    eci = evaluator.get_eci_dict()

    assert len(eci) > 0

    fname = make_tempfile("eci.json")
    evaluator.save_eci(fname=fname)

    # Load the ECI dict
    with open(fname) as file:
        eci_loaded = json.load(file)

    assert eci.keys() == eci_loaded.keys()

    for k, v in eci.items():
        assert pytest.approx(v) == eci_loaded[k]


def test_get_eci_by_size(make_eval):
    evaluator = make_eval(fitting_scheme="l1")
    evaluator.fit()
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
        predict_name = predict_name + dict_eval[tmpi]["name"]
        predict_eci = predict_eci + dict_eval[tmpi]["eci"]
        predict_distance = predict_distance + dict_eval[tmpi]["distance"]

    assert name_list == predict_name
    assert eci_list == predict_eci
    assert distance_list == predict_distance


def test_bkg_ignore_consistency(make_eval_with_bkg):
    """
    Test that ignoring background species is identical to not having
    the sublattice in the first place.
    """

    evaluator_bkg, evaluator_no_bkg = make_eval_with_bkg()
    # Here we test that the two cases generate same basis function and coefficient matrice
    # e.g. the background is completely ignored
    assert (evaluator_bkg.cf_matrix == evaluator_no_bkg.cf_matrix).all()
    assert evaluator_bkg.settings.basis_functions == evaluator_no_bkg.settings.basis_functions

    acon_bkg = evaluator_bkg.atomic_concentrations
    acon_nobkg = evaluator_no_bkg.atomic_concentrations
    for concs_bkg, concs_nobkg in zip(acon_bkg, acon_nobkg):
        concs_bkg.pop("Ag")
        # The atomic density with the background should be half of that with the background removed
        assert {specie: conc * 2 for specie, conc in concs_bkg.items()} == concs_nobkg


def test_custom_prop(bc_setting):
    # Make a database, with some noise in as a custom property
    # we want to fit
    db_name = bc_setting.db_name
    con = connect(db_name)
    expected = []
    for row in con.select(struct_type="final"):
        value = random.random()
        con.update(row.id, dummy=value)
        expected.append(value)
    assert len(expected) > 0  # we should've found some structures
    # Test we can make the evaluate class
    eva = Evaluate(bc_setting, prop="dummy")

    # Currently, the "property" is just "e_dft"
    assert eva.e_dft == pytest.approx(expected)


def test_fit_required(make_eval):
    evaluator: Evaluate = make_eval()
    assert evaluator.fit_required()
    evaluator.fit()
    assert not evaluator.fit_required()
    evaluator.set_fitting_scheme()
    assert evaluator.fit_required()
    evaluator.fit()
    assert not evaluator.fit_required()


def test_load_eci_dict(make_eval):
    evaluator: Evaluate = make_eval()

    dct = {"c0": 1.0, "c3_d0001_0_000": -3.1}
    evaluator.load_eci_dict(dct)

    eci = evaluator.eci
    assert len(evaluator.cf_names) == len(eci)
    assert len(eci) > len(dct)

    for name, value in zip(evaluator.cf_names, eci):
        # Any names not in the dictionary should be 0
        assert value == pytest.approx(dct.get(name, 0.0))


@pytest.mark.parametrize(
    "regressor",
    [
        lambda eva: "l1",
        lambda eva: "l2",
        lambda eva: "ols",
        lambda eva: regression.LinearRegression(),
        lambda eva: regression.Tikhonov(),
        lambda eva: regression.Lasso(),
        lambda eva: regression.BayesianCompressiveSensing(),
        lambda eva: regression.ConstrainedRidge(alpha=np.ones(len(eva.cf_names))),
        lambda eva: regression.BayesianCompressiveSensing(),
        lambda eva: regression.GeneralizedRidgeRegression(np.ones(len(eva.cf_names))),
        lambda eva: regression.PhysicalRidge(cf_names=eva.cf_names),
    ],
)
def test_regressor_fit(regressor, make_eval):
    evaluator: Evaluate = make_eval()
    reg = regressor(evaluator)  # Initialize the regressor
    evaluator.set_fitting_scheme(fitting_scheme=reg)
    evaluator.fit()

    # Test that we can do the basic fit plots. Will not open the figures.
    pp.plot_fit(evaluator)
    pp.plot_fit_residual(evaluator)
    pp.plot_eci(evaluator)


def test_get_eci(make_eval):
    evaluator: Evaluate = make_eval()
    with pytest.raises(ValueError):
        evaluator.get_eci()
    evaluator.fit()
    evaluator.get_eci()


@pytest.mark.parametrize(
    "regressor, expected",
    [
        (lambda: regression.Tikhonov(), True),
        (lambda: regression.Lasso(), True),
        (lambda: regression.LinearRegression(), False),
        (lambda: regression.BayesianCompressiveSensing(), False),
        (lambda: regression.ConstrainedRidge(alpha=np.ones(4)), False),
        (lambda: regression.GeneralizedRidgeRegression(np.ones(4)), False),
    ],
)
def test_supports_alpha_cv(regressor, expected):
    assert supports_alpha_cv(regressor()) is expected


def test_set_scoring_scheme(make_eval):
    eva: Evaluate = make_eval()
    eva.scoring_scheme = "K-FOLD"
    assert eva.scoring_scheme == "k-fold"
    eva.scoring_scheme = "LOOcV"
    assert eva.scoring_scheme == "loocv"
    eva.scoring_scheme = "loocv_fasT"
    assert eva.scoring_scheme == "loocv_fast"
    with pytest.raises(ValueError):
        eva.scoring_scheme = "loo"
