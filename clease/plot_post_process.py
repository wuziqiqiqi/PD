from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from clease import Evaluate, ConvexHull
from clease.evaluate import supports_alpha_cv


def plot_fit(evaluate: Evaluate, plot_args: dict = None, interactive: bool = False) -> Figure:
    """
    Figure object calculated (DFT) and predicted energies.
    If the plot_args dictionary contains keys,
    return  figure object to relate plot_args  keys

    :param evaluate: Use the evaluate object to define the plot argument.
    :param plot_args: plot_args dictionary contains:

        - "xlabel": x-axis label
        - "ylabel": y-axis label
        - "title": title of plot

    :param interactive: Add interactive elements to the plot.

    :return: Figure instance of plot
    """
    if plot_args is None:
        plot_args = {}
    X = evaluate.get_energy_true()
    Y = evaluate.get_energy_predict()
    xlabel = plot_args.get("xlabel", r"E$_{DFT}$ (eV/atom)")
    ylabel = plot_args.get("ylabel", r"E$_{CE}$ (eV/atom)")
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
    cv_name = evaluate.scoring_scheme.lower()

    if cv_name == "k-fold":
        # Figure out the k
        nsplits = evaluate.nsplits
        cv_name = f"{nsplits}-fold CV"

    cv = evaluate.get_cv_score() * 1000
    rmse = evaluate.rmse() * 1000

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.axis([rmin, rmax, rmin, rmax])
    ax.text(
        0.95,
        0.01,
        cv_name + f" = {cv:.3f} meV/atom\n" f"RMSE = {rmse:.3f} meV/atom",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.plot(linear_fit, linear_fit, "r")
    plot_line = ax.plot(X, Y, "bo", mfc="none")[0]

    if interactive:
        # pylint: disable=import-outside-toplevel
        from clease.interactive_plot import ShowStructureOnClick, AnnotatedAx

        annotations = _make_annotations_plot_fit(evaluate)
        # Construct the annotated axis objects.
        annotated_ax = AnnotatedAx(
            ax,
            [plot_line],
            annotations,
            structure_names=[evaluate.names],
        )

        # Attach interactivity to the fig object.
        ShowStructureOnClick(fig, annotated_ax, evaluate.settings.db_name)

    return fig


def plot_fit_residual(
    evaluate: Evaluate, plot_args: dict = None, interactive: bool = False
) -> Figure:
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
    :param interactive: Add interactive elements to the plot.

    :return: Figure instance of plot
    """
    if plot_args is None:
        plot_args = {}
    X = evaluate.get_energy_true()
    Y = evaluate.get_energy_predict() - X  # eV/atom
    Y *= 1000  # meV/atom
    xlabel = plot_args.get("xlabel", "#OCC")
    ylabel = plot_args.get("ylabel", r"$E_{DFT} - E_{pred}$ (meV/atom)")
    title = plot_args.get("title", "Residual (v)")

    gridspec_kw = {"wspace": 0.0, "width_ratios": [5, 1]}
    fig, ax = plt.subplots(ncols=2, sharey=True, gridspec_kw=gridspec_kw)
    ax[0].axhline(0, ls="--")
    plot_line = ax[0].plot(X, Y, "v", mfc="none")[0]
    ax[0].set_xlabel(r"$E_{DFT}$ (eV/atom)")
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)

    hist, bin_edges = np.histogram(Y, bins=30)
    h = bin_edges[1] - bin_edges[0]
    ax[1].barh(bin_edges[:-1], hist, height=h, color="#bdbdbd")
    ax[1].set_xlabel(xlabel)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)

    if interactive:
        # pylint: disable=import-outside-toplevel
        from clease.interactive_plot import ShowStructureOnClick, AnnotatedAx

        annotations = _make_annotations_plot_fit(evaluate)
        # Construct the annotated axis objects.
        annotated_ax = AnnotatedAx(
            ax[0],
            [plot_line],
            annotations,
            structure_names=[evaluate.names],
        )

        # Attach interactivity to the fig object.
        ShowStructureOnClick(fig, annotated_ax, evaluate.settings.db_name)

    return fig


def plot_eci(
    evaluate: Evaluate,
    plot_args: dict = None,
    ignore_sizes=(),
    interactive: bool = False,
) -> Figure:
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
    :param ignore_sizes: list of ints
        Sizes listed in this list will not be plotted.
        E.g. ``ignore_sizes=[0]`` will exclude the 0-body cluster.
        Default is to not ignore any clusters.
    :param interactive: Add interactive elements to the plot.

    :return: Figure instance of plot
    """
    if plot_args is None:
        plot_args = {}
    # eci_by_size dictionary contains eci, name, distance
    eci_by_size = evaluate.get_eci_by_size()
    xlabel = plot_args.get("xlabel", r"Cluster diameter ($n^{th}$ nearest neighbor)")
    ylabel = plot_args.get("ylabel", "ECI (eV/atom)")
    title = plot_args.get("title", "Plot ECI")
    sizes = plot_args.get("sizes", list(eci_by_size.keys()))

    markers = ["o", "v", "x", "D", "^", "h", "s", "p"]
    lines = []
    annotations = []

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0.0, ls="--", color="grey")
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for size in sizes:
        if ignore_sizes and size in ignore_sizes:
            continue
        data = eci_by_size[size]
        # Add 1, as NN starts from 1 and not 0
        X = np.array(data["distance"]) + 1
        Y = data["eci"]
        mrk = markers[size % len(markers)]
        line = ax.plot(X, Y, label=f"{size}-body", marker=mrk, mfc="none", ls="", markersize=8)

        # Make annotations for interactive plots, since we have all the data we need prepared here.
        if interactive:
            lines.append(line[0])
            annot = [
                (
                    f"Size: {size}\nDiameter: {dist:d}\nName: {name}\n"
                    f"Radius: {radius:.3f} Å\nECI: {eci:.4f} eV/atom"
                )
                for dist, name, eci, radius in zip(X, data["name"], Y, data["radius"])
            ]
            annotations.append(annot)
    ax.legend()

    if interactive:
        # pylint: disable=import-outside-toplevel
        from clease.interactive_plot import InteractivePlot, AnnotatedAx

        # Construct the annotated axis objects.
        annotated_ax = AnnotatedAx(
            ax,
            lines,
            annotations,
        )

        # Attach interactivity to the fig object.
        InteractivePlot(fig, annotated_ax)

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
    if not supports_alpha_cv(evaluate.scheme):
        raise ValueError(f"Scheme {evaluate.scheme!r} doesn't support alpha CV.")
    if plot_args is None:
        plot_args = {}
    alpha_cv_data = evaluate.cv_scores
    xlabel = plot_args.get("xlabel", "alpha")
    ylabel = plot_args.get("ylabel", "CV score (meV/atom)")
    title = plot_args.get("title", "CV score vs. alpha")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # this variable is for getting minimum cv
    min_cv = min(alpha_cv_data, key=lambda x: x["cv"])
    X = []
    Y = []
    for data in alpha_cv_data:
        X.append(data["alpha"])
        # Convert from eV/atom to meV/atom
        Y.append(data["cv"] * 1000)
    ax.plot(X, Y)
    ax.text(
        0.65,
        0.01,
        f"min. CV score:\n"
        f"alpha = {min_cv['alpha']:.10f} \n"
        f"CV = {min_cv['cv'] * 1000.0:.3f}"
        f" meV/atom",
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
        fontsize=10,
    )
    # Display the point with the best CV score as a red circle
    idx = np.argmin(Y)
    ax.plot(X[idx], Y[idx], "ro", markersize=5)
    return fig


def plot_convex_hull(evaluate: Evaluate, interactive: bool = False) -> Figure:
    """Plot the convex hull of an evaluate object.

    Args:
        evaluate (Evaluate): The Evaluate object to draw the convex hull from.
        interactive (bool, optional): Plot as an interactive figure?. Defaults to False.
    """
    e_pred = evaluate.get_energy_predict()

    cnv_hull = ConvexHull(evaluate.settings.db_name, select_cond=evaluate.select_cond)

    # Construct the figure object, and plot the DFT points.
    # Also fetch the Line2D objects which are drawn on the figure (i.e., the scatter points)
    # dft_lines is for interactive plotting.
    fig, dft_lines = cnv_hull.plot(return_lines=True)

    # `conc_per_frame` is the concentration with respect to the total number of atoms for
    # each frame
    conc_per_frame = evaluate.atomic_concentrations

    # `concs` is dictionary with the keys of species with value being the
    # concentrations among the frames
    # pylint: disable=protected-access
    concs = {key: [] for key in cnv_hull._unique_elem}
    for frame_conc in conc_per_frame:
        for key, value in concs.items():
            value.append(frame_conc.get(key, 0.0))

    form_en = [cnv_hull.get_formation_energy(c, e) for c, e in zip(conc_per_frame, e_pred.tolist())]
    # Draw the CE energies, and fetch the lines, in case we need them for interactive plotting.
    _, ce_lines = cnv_hull.plot(
        fig=fig,
        concs=concs,
        energies=form_en,
        marker="x",
        return_lines=True,
    )

    fig.suptitle("Convex hull DFT (o), CE (x)")

    if interactive:
        # pylint: disable=import-outside-toplevel
        from clease.interactive_plot import ShowStructureOnClick, AnnotatedAx

        ax_list = fig.get_axes()

        annotations = _make_annotations_hull(evaluate)
        # Construct the annotated axis objects.
        annotated_axes = []
        for ii, ax in enumerate(ax_list):
            data_points = (dft_lines[ii], ce_lines[ii])
            annotated_ax = AnnotatedAx(
                ax,
                data_points,
                annotations,
                structure_names=[evaluate.names, evaluate.names],
            )
            annotated_axes.append(annotated_ax)

        ShowStructureOnClick(fig, annotated_axes, evaluate.settings.db_name)

    return fig


def _make_annotations_hull(evaluate: Evaluate) -> Tuple[List[str], List[str]]:
    """Helper function to make annotations for interactive plots."""
    e_pred = evaluate.get_energy_predict()
    e_dft = evaluate.get_energy_true()

    def format_annotation_dft(idx):
        name = evaluate.names[idx]
        row_id = evaluate.row_ids[idx]
        en = e_dft[idx]
        return f"DB ID: {row_id}\nName: {name}\nE(DFT): {en:.4f} eV/atom"

    def format_annotation_ce(idx):
        name = evaluate.names[idx]
        row_id = evaluate.row_ids[idx]
        e_ce = e_pred[idx]
        return f"DB ID: {row_id}\nName: {name}\nE(CE): {e_ce:.4f} eV/atom"

    N = len(evaluate.names)
    an_dft = [format_annotation_dft(idx) for idx in range(N)]
    an_ce = [format_annotation_ce(idx) for idx in range(N)]
    return an_dft, an_ce


def _make_annotations_plot_fit(evaluate: Evaluate) -> Tuple[List[str], List[str]]:
    """Helper function to make annotations for interactive plots."""
    e_pred = evaluate.get_energy_predict()
    e_dft = evaluate.get_energy_true()

    def format_annotation(idx):
        name = evaluate.names[idx]
        row_id = evaluate.row_ids[idx]
        en = e_dft[idx]
        e_ce = e_pred[idx]
        return f"DB ID: {row_id}\nName: {name}\nE(DFT): {en:.4f} eV/atom\nE(CE): {e_ce:.4f} eV/atom"

    N = len(e_pred)
    return ([format_annotation(idx) for idx in range(N)],)
