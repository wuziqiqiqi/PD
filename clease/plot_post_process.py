import matplotlib.pyplot as plt
import numpy as np
from clease import Evaluate
from matplotlib.figure import Figure


def plot_fit(evaluate: Evaluate, plot_args: dict = None) -> Figure:
    """
    Figure object calculated (DFT) and predicted energies.
    If the plot_args dictionary contains keys,
    return  figure object to relate plot_args  keys

    :param evaluate: Use the evaluate object
    to define the plot argument.
    :param plot_args: plot_args dictionary contains:
        - "xlabel": x-axis label
        - "ylabel": y-axis label
        - "title": title of plot

    :return: Figure instance of plot
    """
    if plot_args is None:
        plot_args = {}
    X = evaluate.get_energy_predict()
    Y = evaluate.e_dft
    xlabel = plot_args.get("xlabel", "E_DFT (eV/atom)")
    ylabel = plot_args.get("ylabel", "E_CE (eV/atom)")
    title = plot_args.get("title", f"Fit using {len(evaluate.e_dft)} data points.")

    # rmin, rmax set the plot range of x, y coordinate
    if np.size(X) and np.size(Y) != 0:
        e_range = max(np.append(X, Y)) - min(np.append(X, Y))
        rmin = min(np.append(X, Y)) - 0.05 * e_range
        rmax = max(np.append(X, Y)) + 0.05 * e_range
    else:
        rmin = -10
        rmax = 10
    linear_fit = np.arange(rmin - 10, rmax + 10, 1)
    cv_name = evaluate.scoring_scheme.upper()
    cv = evaluate.get_cv_score()
    rmse = evaluate.rmse() * 1000

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.axis([rmin, rmax, rmin, rmax])
    ax.text(0.95,
            0.01,
            cv_name + f" = {cv:.3f} meV/atom \n"
            f"RMSE = {rmse:.3f} meV/atom",
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes,
            fontsize=12)
    ax.plot(linear_fit, linear_fit, 'r')
    ax.plot(X, Y, 'bo', mfc='none')

    return fig


def plot_fit_residual(evaluate: Evaluate, plot_args: dict = None) -> Figure:
    """
    Figure object subtracted (DFT) and predicted energies.
    If the plot_args dictionary contains keys,
    return  figure object to relate plot_args  keys

    :param evaluate: Use the evaluate object
    to define the plot argument.
    :param plot_args: plot_args dictionary contains:
        - "xlabel": x-axis label
        - "ylabel": y-axis label
        - "title": title of plot

    :return: Figure instance of plot
    """
    if plot_args is None:
        plot_args = {}
    X = evaluate.e_dft
    Y = evaluate.subtract_predict_dft()
    xlabel = plot_args.get("xlabel", "#OCC")
    ylabel = plot_args.get("ylabel", r"$E_{DFT} - E_{pred}$ (meV/atom)")
    title = plot_args.get("title", "Residual (v)")

    gridspec_kw = {"wspace": 0.0, "width_ratios": [5, 1]}
    fig, ax = plt.subplots(ncols=2, sharey=True, gridspec_kw=gridspec_kw)
    ax[0].set_title(title)
    ax[0].set_ylabel(ylabel)
    ax[0].axhline(0, ls="--")
    ax[0].plot(X, Y, "v", mfc="none")

    hist, bin_edges = np.histogram(Y, bins=30)
    h = bin_edges[1] - bin_edges[0]
    ax[1].barh(bin_edges[:-1], hist, height=h, color="#bdbdbd")
    ax[1].set_xlabel(xlabel)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)

    return fig


def plot_eci(evaluate: Evaluate, plot_args: dict = None) -> Figure:
    """
    Figure object of ECI value according to cluster diameter
    If the plot_args dictionary contains keys,
    return  figure object to relate plot_args  keys

    :param evaluate: Use the evaluate object
    to define the plot argument.
    :param plot_args: plot_args dictionary contains:
        - "xlabel": x-axis label
        - "ylabel": y-axis label
        - "title": title of plot
        - "sizes": list of int to include n-body cluster in plot

    :return: Figure instance of plot
    """
    if plot_args is None:
        plot_args = {}
    # eci_by_size dictionary contains eci, name, distance
    eci_by_size = evaluate.get_eci_by_size()
    xlabel = plot_args.get("xlabel", "Cluster diameter ($n^{th}$ nearest neighbor)")
    ylabel = plot_args.get("ylabel", "ECI (eV/atom)")
    title = plot_args.get("title", "Plot ECI")
    sizes = plot_args.get("sizes", list(eci_by_size.keys()))

    markers = ["o", "v", "x", "D", "^", "h", "s", "p"]
    lines = []

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0.0, ls="--", color="grey")
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for size in sizes:
        data = eci_by_size[size]
        X = data["distance"]
        Y = data["eci"]
        mrk = markers[size % len(markers)]
        line = ax.plot(X, Y, label=f"{size}-body", marker=mrk, mfc="none", ls="", markersize=8)
        lines.append(line[0])
    ax.legend()
    return fig


def plot_cv(evaluate: Evaluate, plot_args: dict = None) -> Figure:
    """
    Figure object of CV values according to alpha values
    If the plot_args dictionary contains keys,
    return  figure object to relate plot_args  keys

    :param evaluate: Use the evaluate object
    to define the plot argument.
    :param plot_args: plot_args dictionary contains:
        - "xlabel": x-axis label
        - "ylabel": y-axis label
        - "title": title of plot

    :return: Figure instance of plot
    """
    if plot_args is None:
        plot_args = {}
    alpha_cv_data = evaluate.cv_scores
    xlabel = plot_args.get("xlabel", "alpha")
    ylabel = plot_args.get("ylabel", 'CV score (meV/atom)')
    title = plot_args.get("title", 'CV score vs. alpha')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # this variable is for getting minimum cv
    min_cv = min(alpha_cv_data, key=lambda x: x['cv'])
    X = []
    Y = []
    for data in alpha_cv_data:
        X.append(data['alpha'])
        Y.append(data['cv'])
    ax.plot(X, Y)
    ax.text(0.65,
            0.01, f"min. CV score:\n"
            f"alpha = {min_cv['alpha']:.10f} \n"
            f"CV = {min_cv['cv'] * 1000.0:.3f}"
            f" meV/atom",
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontsize=10)
    return fig
