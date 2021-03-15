import numpy as np
from clease.plot_post_process import plot_fit, plot_fit_residual,\
    plot_eci, plot_cv

from clease import Evaluate


def test_plot_fit(bc_setting):
    predict = {"title": "plot_FIT_TEST", "xlabel": "E_DFT (eV/atom)", "ylabel": "DFT_FIT"}
    plot_args = {"title": predict["title"], "ylabel": predict["ylabel"]}
    evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
    evaluator.get_eci()

    # x-axis values calculated by get_energy_predict
    predict['x_axis_value'] = \
        np.ndarray.tolist(evaluator.get_energy_predict())
    # y-axis values calculated by get_energy_predict
    predict['y_axis_value'] = \
        np.ndarray.tolist(evaluator.e_dft)

    fig = plot_fit(evaluator, plot_args)
    assert 'loocv' in fig.get_axes()[0].texts[0].get_text()
    assert predict["title"] == fig.get_axes()[0].get_title()
    assert predict["xlabel"] == fig.get_axes()[0].get_xlabel()
    assert predict["ylabel"] == fig.get_axes()[0].get_ylabel()
    assert predict['x_axis_value'] == np.ndarray.tolist(fig.gca().lines[1].get_xdata())
    assert predict['y_axis_value'] == np.ndarray.tolist(fig.gca().lines[1].get_ydata())


def test_plot_fit_residual(bc_setting):
    predict = {"title": "plot_FIT_TEST", "xlabel": "#OCC", "ylabel": "DFT_FIT"}
    plot_args = {"title": predict["title"], "ylabel": predict["ylabel"]}
    evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
    evaluator.get_eci()

    # y-axis values calculated by subtract_predict_dft
    predict['delta_e'] = np.ndarray.tolist(evaluator.subtract_predict_dft())

    fig = plot_fit_residual(evaluator, plot_args)
    assert predict["title"] == fig.get_axes()[0].get_title()
    assert predict["xlabel"] == fig.get_axes()[1].get_xlabel()
    assert predict["ylabel"] == fig.get_axes()[0].get_ylabel()
    assert predict['delta_e'] == np.ndarray.tolist(fig.get_children()[1].get_lines()[1].get_ydata())


def test_plot_eci(bc_setting):
    predict = {
        "title": "plot_FIT_TEST",
        "xlabel": "Cluster diameter ($n^{th}$ nearest neighbor)",
        "ylabel": "DFT_FIT"
    }
    plot_args = {"title": predict["title"], "ylabel": predict["ylabel"]}
    evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
    evaluator.get_eci()

    # x, y-axis values calculated by get_eci_by_size
    test_list = evaluator.get_eci_by_size()
    test_list = list(test_list.values())

    # x, y-axis values of 4 body cluster eci
    assert len(test_list) == 5, len(test_list)
    predict['eci'] = test_list[4]['eci']

    # x, y-axis values of axhline
    predict['axhline_xy'] = [[0.0, 0.0], [1.0, 0.0]]

    fig = plot_eci(evaluator, plot_args)
    assert predict["title"] == fig.get_axes()[0].get_title()
    assert predict["xlabel"] == fig.get_axes()[0].get_xlabel()
    assert predict["ylabel"] == fig.get_axes()[0].get_ylabel()
    assert predict['eci'] == np.ndarray.tolist(fig.gca().lines[5].get_ydata())
    assert predict['axhline_xy'] == np.ndarray.tolist(fig.gca().axhline().get_xydata())


def test_plot_cv(bc_setting):
    predict = {"title": "plot_FIT_TEST", "xlabel": "alpha", "ylabel": "DFT_FIT"}
    plot_args = {"title": predict["title"], "ylabel": predict["ylabel"]}

    evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    evaluator.cv_for_alpha(alphas)

    predict['alpha_cv'] = evaluator.cv_scores

    fig = plot_cv(evaluator, plot_args)

    assert predict['title'] == fig.get_axes()[0].get_title()
    assert predict["xlabel"] == fig.get_axes()[0].get_xlabel()
    assert predict["ylabel"] == fig.get_axes()[0].get_ylabel()

    true_list = fig.gca().get_lines()[0].get_xdata().tolist()

    for predict, true in zip(predict['alpha_cv'], true_list):
        assert predict['alpha'] == true
