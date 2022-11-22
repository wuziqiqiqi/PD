"""Module that fits ECIs to energy data."""
# pylint: skip-file
import os
import sys
import json
import logging as lg
import multiprocessing as mp
from typing import Dict, List, Sequence, Optional
from collections import defaultdict, Counter

from deprecated import deprecated
import numpy as np
from ase.db import connect
import threadpoolctl

from clease.settings import ClusterExpansionSettings
from clease.regression import LinearRegression
from clease.mp_logger import MultiprocessHandler
from clease.tools import singlets2conc, get_ids, get_attribute
from clease.data_manager import make_corr_func_data_manager
from clease.cluster_coverage import ClusterCoverageChecker
from clease.tools import (
    add_file_extension,
    sort_cf_names,
    get_size_from_cf_name,
    get_diameter_from_cf_name,
)

__all__ = ("Evaluate", "supports_alpha_cv")

# Initialize a module-wide logger
logger = lg.getLogger(__name__)


class Evaluate:
    """
    Evaluate RMSE/MAE of the fit and CV scores.

    :param settings: ClusterExpansionSettings object

    :param prop: str
        User defined property for the fit. The property should exist in
        database as key-value pairs. Default is ``energy``.

    :param cf_names: list
        Names of clusters to include in the evalutation.
        If None, all of the possible clusters are included.

    :param select_cond: tuple or list of tuples (optional)
        Custom selection condition specified by user.
        Default only includes "converged=True" and
        "struct_type='initial'".

    :param max_cluster_size: int
        maximum number of atoms in the cluster to include in the fit.
        If ``None``, no restriction on the number of atoms will be imposed.

    :param max_cluster_dia: float or int
        maximum diameter of the cluster (in angstrom) to include in the fit.
        If ``None``, no restriction on the diameter. Note that this diameter of
        the circumscribed sphere, which is slightly different from the meaning
        of max_cluster_dia in `ClusterExpansionSettings` where it refers to the
        the maximum internal distance between any of the atoms in the cluster.

    :param scoring_scheme: str
        should be one of 'loocv', 'loocv_fast' or 'k-fold'

    :param min_weight: float
        Weight given to the data point furthest away from any structure on the
        convex hull. An exponential weighting function is used and the decay
        rate is calculated as

        decay = log(min_weight)/min(sim_measure)

        where sim_measure is a similarity measure used to asses how different
        the structure is from structures on the convex hull.

    :param nsplits: int
        Number of splits to use when partitioning the dataset into
        training and validation data. Only used when scoring_scheme='k-fold'

    :param num_repetitions: int
        Number of repetitions used to use when calculating k-fold cross
        validation. The partitioning is repeated num_repetitions times
        and the resulting value is the average of the k-fold cross
        validation score obtained in each of the runs.
    """

    def __init__(
        self,
        settings,
        prop="energy",
        cf_names=None,
        select_cond=None,
        parallel=False,
        num_core="all",
        fitting_scheme="ridge",
        alpha=1e-5,
        max_cluster_size=None,
        max_cluster_dia=None,
        scoring_scheme="loocv",
        min_weight=1.0,
        nsplits=10,
        num_repetitions=1,
        normalization_symbols: Optional[Sequence[str]] = None,
    ):
        """Initialize the Evaluate class."""
        if not isinstance(settings, ClusterExpansionSettings):
            msg = "settings must be ClusterExpansionSettings object"
            raise TypeError(msg)

        self.settings = settings
        self.prop = prop
        if cf_names is None:
            self.cf_names = sort_cf_names(self.settings.all_cf_names)
        else:
            self.cf_names = sort_cf_names(cf_names)
        self.num_elements = settings.num_elements
        self.scoring_scheme = scoring_scheme

        self.scheme = None
        self.nsplits = nsplits
        self.num_repetitions = num_repetitions

        # Define the selection conditions
        self.select_cond = []
        if select_cond is None:
            self.select_cond = self.default_select_cond
        else:
            if isinstance(select_cond, list):
                self.select_cond += select_cond
            else:
                self.select_cond.append(select_cond)

        # Remove the cluster names that correspond to clusters larger than the
        # specified size and diameter.
        if max_cluster_size is not None:
            self.cf_names = self._filter_cname_on_size(max_cluster_size)
        if max_cluster_dia is not None:
            max_dia = self._get_max_cluster_dia(max_cluster_dia)
            self.cf_names = self._filter_cname_circum_dia(max_dia)

        tab_name = f"{self.settings.basis_func_type.name}_cf"

        # TODO: At a later stage we might want to pass the data manager as an
        # argument since Evaluate does not depend on the details on how the
        # data was optained
        self.dm = make_corr_func_data_manager(self.prop, settings.db_name, tab_name, self.cf_names)

        self.cf_matrix, self.e_dft = self.dm.get_data(self.select_cond)

        self.row_ids = get_ids(self.select_cond, settings.db_name)
        with connect(settings.db_name) as db:
            cur = db.connection.cursor()
            self.names = get_attribute(self.row_ids, cur, "name", "text_key_values")

        self.effective_num_data_pts = len(self.e_dft)
        self.weight_matrix = np.eye(len(self.e_dft))
        self._update_convex_hull_weight(min_weight)

        self.multiplicity_factor = self.settings.multiplicity_factor
        self.eci = None
        self.e_pred_loo = None
        self.parallel = parallel
        if parallel:
            if num_core == "all":
                self.num_core = int(mp.cpu_count() / 2)
            else:
                self.num_core = int(num_core)

        self.set_fitting_scheme(fitting_scheme, alpha)
        self._cv_scores = []

        self.set_normalization(normalization_symbols)

    def set_normalization(self, normalization_symbols: Optional[Sequence[str]] = None) -> None:
        """Set the energy normalization factor, e.g. to normalize the final energy reports
        in energy per metal atom, rather than energy per atom (i.e. every atom).

        :param normalization_symbols: A list of symbols which should be included in the counting.
            If this is None, then the default of normalizing to energy per every atom is maintained.
        """
        self.normalization = np.ones(len(self.e_dft), dtype=float)

        if normalization_symbols is None:
            return
        # We need to figure out the ratio between the total number of atoms, and the number of
        # symbols we normalize to.
        # Energies are assumed to be in energy per atoms, i.e. normalized by the total number of
        # atoms in the initial cell, including vacancies.
        con = self.settings.connect()
        for ii, uid in enumerate(self.row_ids):
            row = con.get(id=uid)
            # Count the occurence of each symbol
            count = Counter(row.symbols)
            natoms = row.natoms

            new_total = sum(count.get(s, 0) for s in normalization_symbols)
            if new_total > 0:
                # If none of the requested species were found we do not adjust the normalization.
                self.normalization[ii] = natoms / new_total

    @property
    def scoring_scheme(self) -> str:
        return self._scoring_scheme

    @scoring_scheme.setter
    def scoring_scheme(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"Scoring scheme should be string, got {value!r}")
        value = value.lower()
        allowed_schemas = {"loocv", "loocv_fast", "k-fold"}
        if value not in allowed_schemas:
            schemas_s = ", ".join(sorted(allowed_schemas))
            raise ValueError(f"Unknown scoring scheme: {value}. Allowed schemas: {schemas_s}")
        self._scoring_scheme = value

    @property
    def concentrations(self):
        """
        The internal concentrations normalised against the 'active' sublattices
        """
        singlet_cols = [i for i, n in enumerate(self.cf_names) if n.startswith("c1")]
        return singlets2conc(self.settings.basis_functions, self.cf_matrix[:, singlet_cols])

    @property
    def default_select_cond(self):
        return [("converged", "=", True), ("struct_type", "=", "initial")]

    def set_fitting_scheme(self, fitting_scheme="ridge", alpha=1e-9):
        from clease.regression import LinearRegression

        allowed_fitting_schemes = ["ridge", "tikhonov", "lasso", "l1", "l2", "ols", "linear"]
        if isinstance(fitting_scheme, LinearRegression):
            self.scheme = fitting_scheme
        elif isinstance(fitting_scheme, str):
            fitting_scheme = fitting_scheme.lower()
            if fitting_scheme not in allowed_fitting_schemes:
                raise ValueError(f"Fitting scheme has to be one of {allowed_fitting_schemes}")
            if fitting_scheme in ["ridge", "tikhonov", "l2"]:
                from clease.regression import Tikhonov

                self.scheme = Tikhonov(alpha=alpha)
            elif fitting_scheme in ["lasso", "l1"]:
                from clease.regression import Lasso

                self.scheme = Lasso(alpha=alpha)
            elif fitting_scheme in ["ols", "linear"]:
                # Perform ordinary least squares
                self.scheme = LinearRegression()
            else:
                raise ValueError(f"Unknown fitting scheme: {fitting_scheme}")
        else:
            raise ValueError(
                f"Fitting scheme has to be one of "
                f"{allowed_fitting_schemes} or a "
                f"LinearRegression instance."
            )

        # Unset the ECI, so a new fit is required.
        self.eci = None

        N = len(self.e_dft)

        # If user has supplied any data weighting, pass the
        # weight matrix to the fitting scheme
        if np.any(np.abs(self.weight_matrix - np.eye(N) > 1e-6)):
            self.scheme.weight_matrix = self.weight_matrix

    def _get_max_cluster_dia(self, max_cluster_dia):
        """Make max_cluster_dia in a numpy array form."""
        if isinstance(max_cluster_dia, (list, np.ndarray)):
            if len(max_cluster_dia) == self.settings.max_cluster_size + 1:
                for i in range(2):
                    max_cluster_dia[i] = 0.0
                max_cluster_dia = np.array(max_cluster_dia, dtype=float)
            elif len(max_cluster_dia) == self.settings.max_cluster_size - 1:
                max_cluster_dia = np.array(max_cluster_dia, dtype=float)
                max_cluster_dia = np.insert(max_cluster_dia, 0, [0.0, 0.0])
            else:
                raise ValueError("Invalid length for max_cluster_dia.")
        # max_cluster_dia is int or float
        elif isinstance(max_cluster_dia, (int, float)):
            max_cluster_dia *= np.ones(self.settings.max_cluster_size - 1, dtype=float)
            max_cluster_dia = np.insert(max_cluster_dia, 0, [0.0, 0.0])

        return max_cluster_dia

    def _update_convex_hull_weight(self, min_weight):
        """Weight structure according to similarity with the
        most similar structure on the Convex Hull."""

        if abs(min_weight - 1.0) < 1e-4:
            return

        from clease import ConvexHull

        cnv_hull = ConvexHull(self.settings.db_name, select_cond=self.select_cond)
        hull = cnv_hull.get_convex_hull()

        cosine_sim = []
        for conc, energy in zip(self.concentrations, self.e_dft):
            sim = cnv_hull.cosine_similarity_convex_hull(conc, energy, hull)
            cosine_sim.append(sim)
        cosine_sim = np.array(cosine_sim)

        # Shift tha maximum value to 0
        cosine_sim -= np.max(cosine_sim)
        min_sim = np.min(cosine_sim)

        decay = np.log(min_weight) / min_sim

        self.weight_matrix = np.diag(np.exp(decay * cosine_sim))
        self.effective_num_data_pts = np.sum(self.weight_matrix)

    def fit_required(self) -> bool:
        """Check whether we need to calculate the ECI values."""
        return self.eci is None

    def fit(self) -> None:
        """Determine the ECI with the given regressor.

        This will always calculate a new fit.
        """
        self.eci = self.scheme.fit(self.cf_matrix, self.e_dft)
        assert self.eci is not None

    def get_eci(self) -> np.ndarray:
        """Determine and return ECIs for a given alpha.
        Raises a ValueError if no fit has been performed yet.

        Returns:
            np.ndarray: A 1D array of floats with all ECI values.
        """
        if self.eci is None:
            # getting ECI's was not allowed to fit, and we haven't run a fit yet.
            raise ValueError("ECI's has not been fit yet. Call the Evaluate.fit method first.")

        return self.eci

    def get_eci_dict(self, cutoff_tol: float = 1e-14) -> Dict[str, float]:
        """Determine cluster names and their corresponding ECI value and return
        them in a dictionary format.

        Args:
            cutoff_tol (float, optional): Cutoff value below which the absolute ECI
                value is considered to be 0. Defaults to 1e-14.

        Returns:
            Dict[str, float]: Dictionary with the CF names and the corresponding
                ECI value.
        """
        eci = self.get_eci()

        # sanity check
        if len(self.cf_names) != len(eci):
            raise ValueError("lengths of cf_names and ECIs are not same")

        all_nonzero = np.abs(eci) > cutoff_tol
        # Only keep the all non-zero values.
        return {cf: val for cf, val, nonzero in zip(self.cf_names, eci, all_nonzero) if nonzero}

    def load_eci_dict(self, eci_dict: Dict[str, float]) -> None:
        """Load the ECI's from a dictionary. Any ECI's which are missing
        from the internal cf_names list are assumed to be 0.

        Note: this doesn't load the scheme or the alpha value, so it will not
        prevent a new fit to be performed if requested, as it may be incompatible
        with the current fitting scheme.
        """

        eci = np.zeros(len(self.cf_names), dtype=float)
        for ii, name in enumerate(self.cf_names):
            eci[ii] = eci_dict.get(name, 0.0)
        self.eci = eci

    def load_eci(self, fname="eci.json") -> None:
        """Read in ECI values stored to a json file.

        Note: this doesn't load the scheme or the alpha value, so it will not
        prevent a new fit to be performed if requested, as it may be incompatible
        with the current fitting scheme.
        """
        full_fname = add_file_extension(fname, ".json")
        with open(full_fname, "r") as fd:
            self.load_eci_dict(json.load(fd))

    def save_eci(self, fname="eci.json", **kwargs):
        """
        Save a dictionary of cluster names and their corresponding ECI value
        in JSON file format.

        Parameters:

        fname: str
            json filename. If no extension if given, .json is added
        kwargs:
            Extra keywords are passed on to the :meth:`~get_eci_dict` method.
        """
        full_fname = add_file_extension(fname, ".json")
        with open(full_fname, "w") as outfile:
            json.dump(self.get_eci_dict(**kwargs), outfile, indent=2, separators=(",", ": "))

    @deprecated(reason="Use the clease.plot_post_process module instead.", version="0.11.7")
    def plot_fit(self, interactive=False, savefig=False, fname=None, show_hull=True):
        """Plot calculated (DFT) and predicted energies for a given alpha.

        Paramters:

        alpha: int or float
            regularization parameter.

        savefig: bool
            - True: Save the plot with a file name specified in 'fname'.
                    Only works when interactive=False.
                    This option does not display figure.
            - False: Display figure without saving.

        fname: str
            file name of the figure (only used when savefig = True)

        show_hull: bool
            whether or not to show convex hull.
        """
        import matplotlib.pyplot as plt
        from clease.interactive_plot import ShowStructureOnClick, AnnotatedAx
        import clease.plot_post_process as pp

        if self.eci is None:
            self.fit()
        e_pred = self.cf_matrix.dot(self.eci)

        e_range = max(np.append(self.e_dft, e_pred)) - min(np.append(self.e_dft, e_pred))
        rmin = min(np.append(self.e_dft, e_pred)) - 0.05 * e_range
        rmax = max(np.append(self.e_dft, e_pred)) + 0.05 * e_range

        prefix = ""
        if fname is not None:
            prefix = fname.rpartition(".")[0]

        cv = None
        cv_name = "LOOCV"
        cv = self.get_cv_score() * 1000.0
        if self.scoring_scheme == "k-fold":
            cv_name = f"{self.nsplits}-fold"

        t = np.arange(rmin - 10, rmax + 10, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if self.effective_num_data_pts != len(self.e_dft):
            ax.set_title(
                f"Fit using {self.e_dft.shape[0]} data points. Eff. "
                f"num. data points {self.effective_num_data_pts:.1f}"
            )
        else:
            ax.set_title(f"Fit using {self.e_dft.shape[0]} data points.")

        if self.effective_num_data_pts != len(self.e_dft):
            w = np.diag(self.weight_matrix)
            im = ax.scatter(e_pred, self.e_dft, c=w)
            cb = fig.colorbar(im)
            cb.set_label("Weight")

            # Plot again with zero marker width to make the interactive
            # plot work
            ax.plot(e_pred, self.e_dft, "o", mfc="none", color="black", markeredgewidth=0.0)
        else:
            ax.plot(e_pred, self.e_dft, "bo", mfc="none")
        ax.plot(t, t, "r")
        ax.axis([rmin, rmax, rmin, rmax])
        ax.set_ylabel(r"$E_{DFT}$ (eV/atom)")
        ax.set_xlabel(r"$E_{pred}$ (eV/atom)")
        ax.text(
            0.95,
            0.01,
            cv_name + f" = {cv:.3f} meV/atom \n" f"RMSE = {self.rmse() * 1000.0:.3f} meV/atom",
            verticalalignment="bottom",
            horizontalalignment="right",
            transform=ax.transAxes,
            fontsize=12,
        )
        if self.e_pred_loo is not None:
            ax.plot(self.e_pred_loo, self.e_dft, "ro", mfc="none")

        if interactive:
            lines = ax.get_lines()
            if self.e_pred_loo is None:
                data_points = [lines[0]]
            else:
                data_points = [lines[0], lines[2]]
            annotations = [self.names, self.names]
            db_name = self.settings.db_name
            annotated_ax = AnnotatedAx(
                ax,
                data_points,
                annotations,
                structure_names=[self.names, self.names],
            )
            ShowStructureOnClick(fig, annotated_ax, db_name)
        else:
            if savefig:
                plt.savefig(fname=fname)
            else:
                plt.show()

        # Create a plot with the residuals

        gridspec_kw = {"wspace": 0.0, "width_ratios": [5, 1]}

        fig_residual, ax_res = plt.subplots(ncols=2, sharey=True, gridspec_kw=gridspec_kw)
        ax_residual = ax_res[0]
        ax_residual.set_title("LOO residual (o). Residual (v)")
        if self.e_pred_loo is None:
            loo_delta = None
        else:
            loo_delta = (self.e_dft - self.e_pred_loo) * 1000.0
        delta_e = (self.e_dft - e_pred) * 1000.0
        if self.effective_num_data_pts != len(self.e_dft) and loo_delta is not None:
            im = ax_residual.scatter(self.e_dft, loo_delta, c=w)
            cb = fig_residual.colorbar(im)
            cb.set_label("Weight")

            # Plot again with zero with to make the interactive
            # plot work
            ax_residual.plot(
                self.e_dft,
                loo_delta,
                "o",
                color="black",
                markeredgewidth=0.0,
                mfc="none",
            )
        else:
            if loo_delta is not None:
                ax_residual.plot(self.e_dft, loo_delta, "o")

        ax_residual.plot(self.e_dft, delta_e, "v", mfc="none")

        ax_residual.axhline(0, ls="--")
        ax_residual.set_ylabel(r"$E_{DFT} - E_{pred}$ (meV/atom)")

        hist, bin_edges = np.histogram(delta_e, bins=30)
        h = bin_edges[1] - bin_edges[0]
        ax_res[1].barh(bin_edges[:-1], hist, height=h, color="#bdbdbd")

        ax_res[1].set_xlabel("# occ")
        ax_res[1].spines["right"].set_visible(False)
        ax_res[1].spines["top"].set_visible(False)

        if interactive:
            lines = ax_residual.get_lines()
            if loo_delta is not None:
                data_points = [lines[0], lines[1]]
                annotations = [self.names, self.names]
            else:
                data_points = [lines[0]]
                annotations = [self.names]

            annotated_ax = AnnotatedAx(ax_residual, data_points, annotations)
            ShowStructureOnClick(fig_residual, annotated_ax, db_name)
        else:
            if savefig:
                fig_residual.savefig(prefix + "_residuals.png")
            else:
                plt.show()

        # Optionally show the convex hull
        if show_hull:
            fig = pp.plot_convex_hull(evaluate=self, interactive=interactive)

            if interactive:
                # Interactive (currently) has a built-in plt.show()
                pass
            else:
                if savefig:
                    fig.savefig(prefix + "_cnv_hull.png")
                else:
                    plt.show()

    @property
    def atomic_concentrations(self):
        """
        The actual atomic concentration (including background lattices) normalised against the
        total number of atoms
        """
        conc_per_frame = []
        conc_ratio = self.settings.atomic_concentration_ratio
        ignored = self.settings.ignored_species_and_conc
        for internal_conc in self.concentrations:
            frame_conc = {}
            # Internal concentrations need to be rescaled
            for specie, cvalue in internal_conc.items():
                frame_conc[specie] = cvalue * conc_ratio
            # Include the ignored species
            for specie, cvalue in ignored.items():
                if specie in frame_conc:
                    frame_conc[specie] += cvalue
                else:
                    frame_conc[specie] = cvalue
            conc_per_frame.append(frame_conc)
        return conc_per_frame

    def alpha_CV(
        self,
        alpha_min=1e-7,
        alpha_max=1.0,
        num_alpha=10,
        scale="log",
        logfile=None,
        fitting_schemes=None,
    ):
        """Calculate CV for a given range of alpha.

        In addition to calculating CV with respect to alpha, a logfile can be
        used to extend the range of alpha or to add more alpha values in a
        given range.

        Returns a list of alpha values, and a list of CV scores.

        Parameters:

        alpha_min: int or float
            minimum value of regularization parameter alpha.

        alpha_max: int or float
            maximum value of regularization parameter alpha.

        num_alpha: int
            number of alpha values to be used in the plot.

        scale: str
            - 'log'(default): alpha values are evenly spaced on a log scale.
            - 'linear': alpha values are evenly spaced on a linear scale.

        logfile: file object, str or None.
            - None: logging is disabled
            - str: a file with that name will be opened. If '-', stdout used.
            - file object: use the file object for logging

        fitting_schemes: None or array of instance of LinearRegression.

        Note: If the file with the same name exists, it first checks if the
              alpha value already exists in the logfile and evalutes the CV of
              the alpha values that are absent. The newly evaluated CVs are
              appended to the existing file.
        """
        from clease.regression import LinearRegression

        if not supports_alpha_cv(self.scheme):
            raise ValueError(f"Scheme must support scalar alpha, got {self.scheme!r}")

        fitting_schemes = self.scheme.get_instance_array(
            alpha_min, alpha_max, num_alpha=num_alpha, scale=scale
        )

        for scheme in fitting_schemes:
            if not isinstance(scheme, LinearRegression):
                raise TypeError(
                    "Each entry in fitting_schemes should be an " "instance of LinearRegression"
                )
            elif not scheme.is_scalar():
                raise TypeError(
                    "plot_CV only supports the fitting schemes " "with a scalar paramater."
                )

        # if the file exists, read the alpha values that are already evaluated.
        self._initialize_logfile(logfile)
        fitting_schemes = self._remove_existing_alphas(logfile, fitting_schemes)

        for scheme in fitting_schemes:
            if not isinstance(scheme, LinearRegression):
                raise TypeError(
                    "Each entry in fitting_schemes should be an " "instance of LinearRegression"
                )
            elif not scheme.is_scalar():
                raise TypeError(
                    "plot_CV only supports the fitting schemes " "with a scalar paramater."
                )

        # if the file exists, read the alpha values that are already evaluated.
        self._initialize_logfile(logfile)
        fitting_schemes = self._remove_existing_alphas(logfile, fitting_schemes)

        # get CV scores
        alphas = []
        if self.parallel:
            # We need to limit NumPy's parallelization (and any other BLAS/OpenMP threading)
            # as it'll spawn num_score * NUM_THREADS threads, which ultimately hurts the performance.
            # We un-limit the threading again after the work is done.
            with threadpoolctl.threadpool_limits(limits=1):
                # Use a context manager to ensure workers are properly closed, even upon a crash
                with mp.Pool(self.num_core) as workers:
                    args = [(self, scheme) for scheme in fitting_schemes]
                    alphas = [s.get_scalar_parameter() for s in fitting_schemes]
                    cv = workers.map(loocv_mp, args)
                    cv = np.array(cv)
        else:
            cv = np.ones(len(fitting_schemes))
            for i, scheme in enumerate(fitting_schemes):
                self.set_fitting_scheme(fitting_scheme=scheme)
                cv[i] = self.get_cv_score()
                self.fit()
                num_eci = len(np.nonzero(self.get_eci())[0])
                alpha = scheme.get_scalar_parameter()
                alphas.append(alpha)
                logger.info(f"{alpha:.10f}\t {num_eci}\t {cv[i]:.10f}")

        # add the cv scores
        self._cv_scores = []
        for alpha, cv_score in zip(alphas, cv):
            self._cv_scores.append({"alpha": alpha, "cv": cv_score})

        return alphas, cv

    def cv_for_alpha(self, alphas: List[float]) -> None:
        """
        Calculate the CV scores for alphas using the fitting scheme
        specified in the Evaluate object.

        :param alphas: List of alpha values to get CV scores
        """
        if not isinstance(self.scheme, LinearRegression):
            raise TypeError("Fitting scheme must be a LinearRegression")

        for alpha in alphas:
            self.scheme.alpha = alpha
            cv = self.get_cv_score()
            self.fit()
            num_eci = len(np.nonzero(self.get_eci())[0])
            self._cv_scores.append({"alpha": alpha, "cv": cv})
            logger.info(f"{alpha:.10f}\t {num_eci}\t {cv:.10f}")

    @property
    def cv_scores(self):
        if not self._cv_scores:
            raise ValueError("CV scores have not been calculated yet")

        return self._cv_scores

    @deprecated(reason="Use the alpha_CV method instead.", version="0.11.7")
    def plot_CV(
        self,
        alpha_min=1e-7,
        alpha_max=1.0,
        num_alpha=10,
        scale="log",
        logfile=None,
        fitting_schemes=None,
        savefig=False,
        fname=None,
    ):
        """Plot CV for a given range of alpha.

        In addition to plotting CV with respect to alpha, logfile can be used
        to extend the range of alpha or add more alpha values in a given range.
        Returns an alpha value that leads to the minimum CV score within the
        pool of evaluated alpha values.

        Parameters:

        alpha_min: int or float
            minimum value of regularization parameter alpha.

        alpha_max: int or float
            maximum value of regularization parameter alpha.

        num_alpha: int
            number of alpha values to be used in the plot.

        scale: str
            - 'log'(default): alpha values are evenly spaced on a log scale.
            - 'linear': alpha values are evenly spaced on a linear scale.

        logfile: file object, str or None
            - None: logging is disabled
            - str: a file with that name will be opened. If '-', stdout used.
            - file object: use the file object for logging

        fitting_schemes: None or array of instance of LinearRegression

        savefig: bool
            - True: Save the plot with a file name specified in 'fname'. This
                    option does not display figure.
            - False: Display figure without saving.

        fname: str
            file name of the figure (only used when savefig = True)

        Note: If the file with the same name exists, it first checks if the
              alpha value already exists in the logfile and evalutes the CV of
              the alpha values that are absent. The newly evaluated CVs are
              appended to the existing file.
        """
        import matplotlib.pyplot as plt

        alphas, cv = self.alpha_CV(
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            num_alpha=num_alpha,
            scale=scale,
            logfile=logfile,
            fitting_schemes=fitting_schemes,
        )
        # --------------- #
        # Generate a plot #
        # --------------- #
        # if logfile is present, read all entries from the file
        if logfile is not None and logfile != "-":
            alphas, cv = self._get_alphas_cv_from_file(logfile)

        # get the minimum CV score and the corresponding alpha value
        ind = cv.argmin()
        min_alpha = alphas[ind]
        min_cv = cv[ind]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("CV score vs. alpha")
        ax.semilogx(alphas, cv * 1000)
        ax.semilogx(min_alpha, min_cv * 1000, "bo", mfc="none")
        ax.set_ylabel("CV score (meV/atom)")
        ax.set_xlabel("alpha")
        ax.text(
            0.65,
            0.01,
            f"min. CV score:\n"
            f"alpha = {min_alpha:.10f} \n"
            f"CV = {min_cv * 1000.0:.3f} meV/atom",
            verticalalignment="bottom",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontsize=10,
        )
        if savefig:
            plt.savefig(fname=fname)
        else:
            plt.show()
        return min_alpha

    @deprecated(reason="Logfile is being removed.", version="0.11.7")
    def _get_alphas_cv_from_file(self, logfile):
        alphas = []
        cv = []
        with open(logfile) as log:
            next(log)
            for line in log:
                alphas.append(float(line.split()[0]))
                cv.append(float(line.split()[-1]))
            alphas = np.array(alphas)
            cv = np.array(cv)
            # sort alphas and cv based on the values of alphas
            ind = alphas.argsort()
            alphas = alphas[ind]
            cv = cv[ind]
        return alphas, cv

    def _remove_existing_alphas(self, logfile, fitting_schemes):
        if not isinstance(logfile, str):
            return fitting_schemes
        elif logfile == "-":
            return fitting_schemes

        existing_alpha = []
        with open(logfile) as f:
            lines = f.readlines()
        for line_num, line in enumerate(lines):
            if line_num == 0:
                continue
            existing_alpha.append(float(line.split()[0]))
        schemes = []
        for scheme in fitting_schemes:
            exists = np.isclose(existing_alpha, scheme.get_scalar_parameter(), atol=1e-9).any()
            if not exists:
                schemes.append(scheme)
        return schemes

    def _initialize_logfile(self, logfile):
        # logfile setup
        if isinstance(logfile, str):
            if logfile == "-":
                handler = lg.StreamHandler(sys.stdout)
                handler.setLevel(lg.INFO)
                logger.addHandler(handler)
            else:
                handler = MultiprocessHandler(logfile)
                handler.setLevel(lg.INFO)
                logger.addHandler(handler)
                # create a log file and make a header line if the file does not
                # exist.
                if os.stat(logfile).st_size == 0:
                    logger.info("alpha \t\t # ECI \t CV")

    @deprecated(reason="Use the clease.plot_post_process module instead.", version="0.11.7")
    def plot_ECI(self, ignore_sizes=(0,), interactive=True):
        """Plot the all the ECI.

        Parameters:

        ignore_sizes: list of ints
            Sizes listed in this list will not be plotted.
            Default is to ignore the emptry cluster.

        interactive: bool
            If ``True``, one can interact with the plot using mouse.
        """
        import matplotlib.pyplot as plt
        from clease.interactive_plot import InteractivePlot, AnnotatedAx

        if self.eci is None:
            self.fit()

        # Structure the ECIs in terms by size
        eci_by_size = {}
        for name, eci in zip(self.cf_names, self.eci):
            size = get_size_from_cf_name(name)
            d = get_diameter_from_cf_name(name)
            if size not in eci_by_size.keys():
                eci_by_size[size] = {"d": [], "eci": [], "name": []}
            eci_by_size[size]["d"].append(d)
            eci_by_size[size]["eci"].append(eci)
            eci_by_size[size]["name"].append(name)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.axhline(0.0, ls="--", color="grey")
        markers = ["o", "v", "x", "D", "^", "h", "s", "p"]
        annotations = []
        lines = []
        for size, data in eci_by_size.items():
            if size in ignore_sizes:
                continue
            data["d"] = np.array(data["d"])
            data["eci"] = np.array(data["eci"])
            sort_index = np.argsort(data["d"])
            data["d"] = data["d"][sort_index]
            data["eci"] = data["eci"][sort_index]
            annotations.append([data["name"][indx] for indx in sort_index])
            mrk = markers[size % len(markers)]
            line = ax.plot(
                data["d"],
                data["eci"],
                label=f"{size}-body",
                marker=mrk,
                mfc="none",
                ls="",
                markersize=8,
            )
            lines.append(line[0])

        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_xlabel("Cluster diameter ($n^{th}$ nearest neighbor)")
        ax.set_ylabel("ECI (eV/atom)")
        ax.legend()
        if interactive:
            annotated_ax = AnnotatedAx(ax, lines, annotations)
            # Note: Internally this calls plt.show()
            InteractivePlot(fig, annotated_ax)
        else:
            plt.show()

    def _get_cf_name_radius(self, cf_name: str) -> float:
        """Get the cluster radius of a cf_name"""
        cluster = self.settings.get_cluster_corresponding_to_cf_name(cf_name)
        return cluster.diameter / 2

    def mae(self):
        """Calculate mean absolute error (MAE) of the fit."""
        if self.eci is None:
            self.fit()
        e_pred = self.cf_matrix.dot(self.eci)
        delta_e = self.e_dft - e_pred
        delta_e *= self.normalization

        w = np.diag(self.weight_matrix)
        delta_e *= w
        return sum(np.absolute(delta_e)) / self.effective_num_data_pts

    def rmse(self):
        """Calculate root-mean-square error (RMSE) of the fit."""
        if self.eci is None:
            self.fit()
        e_pred = self.cf_matrix.dot(self.eci)
        delta_e = self.e_dft - e_pred
        delta_e *= self.normalization

        w = np.diag(self.weight_matrix)
        rmse_sq = np.sum(w * delta_e**2)
        rmse_sq /= self.effective_num_data_pts
        return np.sqrt(rmse_sq)

    def loocv_fast(self):
        """CV score based on the method in J. Phase Equilib. 23, 348 (2002).

        This method has a computational complexity of order n^1.
        """
        # For each structure i, predict energy based on the ECIs determined
        # using (N-1) structures and the parameters corresponding to the
        # structure i.
        # CV^2 = N^{-1} * Sum((E_DFT-E_pred) / (1 - X_i (X^T X)^{-1} X_u^T))^2
        if not self.scheme.support_fast_loocv:
            return self.loocv()

        if self.eci is None:
            self.fit()
        e_pred = self.cf_matrix.dot(self.eci)
        delta_e = self.e_dft - e_pred
        delta_e *= self.normalization

        cfm = self.cf_matrix
        # precision matrix
        prec = self.scheme.precision_matrix(cfm)
        delta_e_loo = delta_e / (1 - np.diag(cfm.dot(prec).dot(cfm.T)))
        self.e_pred_loo = self.e_dft - delta_e_loo

        # Apply energy normalization
        self.e_pred_loo *= self.normalization

        w = np.diag(self.weight_matrix)
        cv_sq = np.sum(w * delta_e_loo**2)

        cv_sq /= self.effective_num_data_pts
        return np.sqrt(cv_sq)

    def loocv(self):
        """Determine the CV score for the Leave-One-Out case."""
        cv_sq = 0.0
        e_pred_loo = []
        for i in range(self.cf_matrix.shape[0]):
            eci = self._get_eci_loo(i)
            e_pred = self.cf_matrix[i][:].dot(eci)
            delta_e = self.e_dft[i] - e_pred
            delta_e *= self.normalization[i]

            cv_sq += self.weight_matrix[i, i] * (delta_e) ** 2
            e_pred_loo.append(e_pred)
        # cv_sq /= self.cf_matrix.shape[0]
        cv_sq /= self.effective_num_data_pts
        self.e_pred_loo = e_pred_loo
        return np.sqrt(cv_sq)

    def k_fold_cv(self):
        """Determine the k-fold cross validation."""
        from clease.tools import split_dataset

        avg_score = 0.0
        for _ in range(self.num_repetitions):
            partitions = split_dataset(self.cf_matrix, self.e_dft, nsplits=self.nsplits)
            scores = []
            for part in partitions:
                eci = self.scheme.fit(part["train_X"], part["train_y"])
                e_pred = part["validate_X"].dot(eci)
                delta_e = e_pred - part["validate_y"]
                delta_e *= self.normalization[part["validate_index"]]

                scores.append(np.mean(delta_e**2))
            avg_score += np.sqrt(np.mean(scores))
        return avg_score / self.num_repetitions

    def _get_eci_loo(self, i):
        """Determine ECI values for the Leave-One-Out case.

        Eliminate the ith row of the cf_matrix when determining the ECIs.
        Returns the determined ECIs.

        Parameters:

        i: int
            iterator passed from the self.loocv method.
        """
        cfm = np.delete(self.cf_matrix, i, 0)
        e_dft = np.delete(self.e_dft, i, 0)
        eci = self.scheme.fit(cfm, e_dft)
        return eci

    def _filter_cname_on_size(self, max_size):
        """
        Filter the cluster names based on size

        Parameters:

        max_size: int
            Maximum cluster size to include in fit
        """
        filtered_names = []
        for name in self.cf_names:
            size = get_size_from_cf_name(name)
            if size <= max_size:
                filtered_names.append(name)
        return filtered_names

    def _filter_cname_circum_dia(self, max_dia):
        """
        Filter the cluster names based on the diameter of the circumscribed
        sphere.

        :param max_dia: list of float
            Diameter of the cirscumscribed sphere for each size.
            Note: Index 0 corresponds to the diameter of 2-body clusters,
            index 1 to 3-body, etc.
        """
        filtered_names = []
        # Note: max_dia is a list starting from 2-body
        max_size_expected = len(max_dia) + 1
        for name in self.cf_names:
            size = get_size_from_cf_name(name)
            if size > max_size_expected:
                raise ValueError(
                    "Inconsistent length of max_dia, size: {}, expected at most: {}".format(
                        size, max_size_expected
                    )
                )
            elif size in [0, 1]:
                filtered_names.append(name)
                continue

            prefix = name.rpartition("_")[0]
            cluster = self.settings.cluster_list.get_by_name(prefix)[0]
            dia = cluster.diameter

            # Index of size in the max_dia array
            size_index = size - 2
            if dia < max_dia[size_index]:
                filtered_names.append(name)
        return filtered_names

    def generalization_error(self, validation_id: List[int]):
        """
        Estimate the generalization error to new datapoints

        :param validation_ids: List with IDs to leave out of the dataset
        """

        db = connect(self.settings.db_name)

        mse = 0.0
        count = 0
        for uid in validation_id:
            row = db.get(id=uid)
            cf = self._get_cf_from_atoms_row(row)

            pred = self.get_eci().dot(cf)

            if row.get("final_struct_id", -1) != -1:
                energy = db.get(id=row.get("final_struct_id"))
            else:
                energy = row.energy
            mse += (energy / row.natoms - pred) ** 2
            count += 1

        if count == 0:
            return 0.0
        return np.sqrt(mse / count) * 1000.0

    def export_dataset(self, fname):
        """
        Export the dataset used to fit a model y = Xc where y is typically the
        DFT energy per atom and c is the unknown ECIs. This function exports
        the data to a csv file with the following format

        # ECIname_1, ECIname_2, ..., ECIname_n, E_DFT
        0.1, 0.4, ..., -0.6, -2.0
        0.3, 0.2, ..., -0.9, -2.3

        thus each row in the file contains the correlation function values and
        the corresponding DFT energy value.

        Parameter:

        fname: str
            Filename to write to. Typically this should end with .csv
        """
        self.dm.to_csv(fname)

    def get_cv_score(self):
        """
        Calculate the CV score according to the selected scheme
        """
        if self.scoring_scheme == "loocv":
            cv = self.loocv()
        elif self.scoring_scheme == "loocv_fast":
            cv = self.loocv_fast()
        elif self.scoring_scheme == "k-fold":
            cv = self.k_fold_cv()
        else:
            raise ValueError(f"Unknown scoring scheme: {self.schoring_scheme}")
        return cv

    def get_energy_predict(self, normalize: bool = True) -> np.ndarray:
        """
        Perform matrix multiplication of eci and cf_matrix

        :return: Energy predicted using ECIs
        """
        eci = self.get_eci()
        en = self.cf_matrix.dot(eci)
        if normalize:
            return en * self.normalization
        return en

    def get_energy_true(self, normalize: bool = True) -> np.ndarray:
        if normalize:
            return self.e_dft * self.normalization
        return self.e_dft

    def get_eci_by_size(self) -> Dict[str, Dict[str, list]]:
        """
        Classify distance, eci and cf_name according to cluster body size

        :return: Dictionary which contains

            * Key: body size of cluster
            * Value: A dictionary with the following entries:

                - "distance" : distance of the cluster
                - "eci" : eci of the cluster
                - "name" : name of the cluster
                - "radius" : Radius of the cluster in Ångstrom.
        """
        if self.eci is None:
            raise ValueError("ECIs have not been fitted yet.")

        # Structure the ECIs in terms by size
        eci_by_size = defaultdict(lambda: defaultdict(list))
        for name, eci in zip(self.cf_names, self.eci):
            size = get_size_from_cf_name(name)
            distance = get_diameter_from_cf_name(name)
            radius = self._get_cf_name_radius(name)

            eci_by_size[size]["distance"].append(distance)
            eci_by_size[size]["eci"].append(eci)
            eci_by_size[size]["name"].append(name)
            eci_by_size[size]["radius"].append(radius)

        # Remove the defaultdict factory attributes
        return {k: dict(v) for k, v in eci_by_size.items()}

    def print_coverage_report(self, file=sys.stdout) -> None:
        """
        Prints a report of how large fraction of the possible variation in each
        cluster is covered by the dataset

        :param file: a file-like object (stream); defaults to the current sys.stdout.
        """
        cov_checker = ClusterCoverageChecker(self.settings, select_cond=self.select_cond)
        cov_checker.print_report(file=file)


def loocv_mp(args):
    """Need to wrap this function in order to use it with multiprocessing.

    Parameters:

    args: Tuple where the first entry is an instance of Evaluate
        and the second is the penalization value
    """
    evaluator = args[0]
    scheme = args[1]
    evaluator.set_fitting_scheme(fitting_scheme=scheme)
    alpha = scheme.get_scalar_parameter()

    if evaluator.scoring_scheme == "loocv":
        cv = evaluator.loocv()
    elif evaluator.scoring_scheme == "loocv_fast":
        cv = evaluator.loocv_fast()
    elif evaluator.scoring_scheme == "k-fold":
        cv = evaluator.k_fold_cv()
    evaluator.fit()
    num_eci = len(np.nonzero(evaluator.get_eci())[0])
    logger.info("%.10f\t %d\t %.10f", alpha, num_eci, cv)
    return cv


def supports_alpha_cv(scheme: LinearRegression) -> bool:
    """Determine whether a regression scheme supports alpha CV"""
    if scheme.is_scalar() and hasattr(scheme, "alpha"):
        return True
    return False
