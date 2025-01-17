import sys
import pytest
import numpy as np
import clease.plot_post_process as pp

from clease import Evaluate


def test_plot_fit(bc_setting):
    predict = {
        "title": "plot_FIT_TEST",
        "xlabel": r"E$_{DFT}$ (eV/atom)",
        "ylabel": r"E$_{CE}$ (eV/atom)",
    }
    plot_args = {"title": predict["title"], "ylabel": predict["ylabel"]}
    evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1e-6)
    evaluator.fit()

    # x-axis is the DFT energy
    predict["x_axis_value"] = evaluator.e_dft
    # y-axis is the CE energy
    predict["y_axis_value"] = evaluator.get_energy_predict()

    fig = pp.plot_fit(evaluator, plot_args)
    assert "loocv" in fig.get_axes()[0].texts[0].get_text()
    assert predict["title"] == fig.get_axes()[0].get_title()
    assert predict["xlabel"] == fig.get_axes()[0].get_xlabel()
    assert predict["ylabel"] == fig.get_axes()[0].get_ylabel()
    assert predict["x_axis_value"] == pytest.approx(fig.gca().lines[1].get_xdata())
    assert predict["y_axis_value"] == pytest.approx(fig.gca().lines[1].get_ydata())


def test_plot_fit_residual(bc_setting):
    predict = {"title": "plot_FIT_TEST", "xlabel": "#OCC", "ylabel": "DFT_FIT"}
    plot_args = {"title": predict["title"], "ylabel": predict["ylabel"]}
    evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1e-6)
    evaluator.fit()

    delta_e = evaluator.get_energy_predict() - evaluator.e_dft
    predict["delta_e"] = delta_e * 1000  # convert to meV/atom

    fig = pp.plot_fit_residual(evaluator, plot_args)
    assert predict["title"] == fig.get_axes()[0].get_title()
    assert predict["xlabel"] == fig.get_axes()[1].get_xlabel()
    assert predict["ylabel"] == fig.get_axes()[0].get_ylabel()
    assert np.allclose(predict["delta_e"], fig.get_children()[1].get_lines()[1].get_ydata())


@pytest.mark.parametrize("ignore_sizes", [(), [0], [0, 1], [3, 4]])
def test_plot_eci(bc_setting, ignore_sizes):
    predict = {
        "title": "plot_FIT_TEST",
        "xlabel": "Cluster diameter ($n^{th}$ nearest neighbor)",
        "ylabel": "DFT_FIT",
    }
    plot_args = {"title": predict["title"], "ylabel": predict["ylabel"]}
    evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1e-6)
    evaluator.fit()

    # x, y-axis values calculated by get_eci_by_size (or largest we keep)
    sizes = evaluator.get_eci_by_size()
    test_list = [v for s, v in sizes.items() if s not in ignore_sizes]

    # Verify we get 5 eci's (0, 1, 2, 3, 4) except the ones we should ignore.
    assert len(test_list) == (5 - len(ignore_sizes)), len(test_list)

    # x, y-axis values of axhline
    predict["axhline_xy"] = [[0.0, 0.0], [1.0, 0.0]]

    fig = pp.plot_eci(evaluator, plot_args, ignore_sizes=ignore_sizes)
    assert predict["title"] == fig.get_axes()[0].get_title()
    assert predict["xlabel"] == fig.get_axes()[0].get_xlabel()
    assert predict["ylabel"] == fig.get_axes()[0].get_ylabel()
    ax = fig.gca()
    # First the h-line is plotted, and then all of the ECI's
    eci_lines = ax.lines[1:]
    for ii, test in enumerate(test_list):
        eci = test["eci"]
        assert np.allclose(eci, eci_lines[ii].get_ydata())
    assert np.allclose(predict["axhline_xy"], ax.axhline().get_xydata())


def test_plot_cv(bc_setting):
    predict = {"title": "plot_FIT_TEST", "xlabel": "alpha", "ylabel": "DFT_FIT"}
    plot_args = {"title": predict["title"], "ylabel": predict["ylabel"]}

    evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1e-6)
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    evaluator.cv_for_alpha(alphas)

    predict["alpha_cv"] = evaluator.cv_scores

    fig = pp.plot_cv(evaluator, plot_args)

    assert predict["title"] == fig.get_axes()[0].get_title()
    assert predict["xlabel"] == fig.get_axes()[0].get_xlabel()
    assert predict["ylabel"] == fig.get_axes()[0].get_ylabel()

    true_list = fig.gca().get_lines()[0].get_xdata().tolist()

    for predict, true in zip(predict["alpha_cv"], true_list):
        assert predict["alpha"] == true


@pytest.mark.parametrize("interactive", [True, False])
def test_plot_ch(interactive, bc_setting):
    evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1e-6)
    evaluator.fit()

    # Simply verify that we can run the plot convex hull plotting function.
    pp.plot_convex_hull(evaluator, interactive=interactive)


@pytest.mark.parametrize(
    "plot_name", ["plot_convex_hull", "plot_fit", "plot_fit_residual", "plot_eci"]
)
def test_plot_interactive_events(bc_setting, plot_name):
    evaluator = Evaluate(bc_setting, fitting_scheme="l2", alpha=1e-6)
    evaluator.fit()

    # Simply verify that we can run the plot convex hull plotting function.
    fnc = getattr(pp, plot_name)
    fig1 = fnc(evaluator, interactive=False)
    fig2 = fnc(evaluator, interactive=True)

    # Ensure there are more events in the interactive one (the ones we added)
    def get_events(fig, event_name):
        return fig.canvas.callbacks.callbacks.get(event_name, {})

    for event in ["button_press_event", "motion_notify_event"]:
        events1 = get_events(fig1, event)
        events2 = get_events(fig2, event)
        # We should have 1 more event
        # Since the object fell out of scope, and the event is only weak-ref'd,
        # it is normally garbage collected, unless we wrap it.
        # If more events are added in the future, the number of expected extra events should be
        # adjusted accordingly.
        if plot_name in {"plot_eci"} and event == "button_press_event":
            # Doesn't add any extra click events
            assert len(events1) == len(events2)
        else:
            assert len(events2) == len(events1) + 1
