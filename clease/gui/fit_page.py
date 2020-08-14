from kivy.uix.popup import Popup
from clease.gui.load_save_dialog import LoadDialog
from clease import Evaluate
from clease.regression import GAFit, LinearRegression
from kivy.app import App
import json
from clease.gui.util import (parse_max_cluster_dia, parse_select_cond)
from clease.gui.constants import FOREGROUND_TEXT_COLOR
from clease.gui.constants import ECI_GRAPH_COLORS
from clease.gui.fitting_alg_editors import (FitAlgEditor, LassoEditor, L2Editor, BCSEditor,
                                            GAEditor)
from clease.gui.legend import Legend
from clease.gui import backup_folder

from threading import Thread
from kivy.uix.screenmanager import Screen
from kivy.utils import get_color_from_hex
from kivy_garden.graph import Graph, ScatterPlot, BarPlot, LinePlot
import numpy as np
import traceback


class ECIOptimizer(object):
    fit_page = None
    evaluator = None

    def optimize(self):
        try:
            self.fit_page.eci = self.evaluator.get_eci_dict()

            e_ce = self.evaluator.cf_matrix.dot(self.evaluator.eci)
            self.fit_page.e_ce = e_ce
            self.fit_page.e_dft = self.evaluator.e_dft
            self.fit_page.e_pred_leave_out = self.evaluator.e_pred_loo
            cv = self.evaluator.get_cv_score()
            rmse = self.evaluator.rmse() * 1000.0
            mae = self.evaluator.mae() * 1000.0
            self.fit_page.set_cv(cv)
            self.fit_page.set_rmse(rmse)
            self.fit_page.set_mae(mae)

            App.get_running_app().root.ids.status.text = 'Idle'
            self.fit_page.fitting_in_progress = False
            self.fit_page.update_plots()
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
            self.fit_page.fitting_in_progress = False
            return


class GAClusterSelector(object):
    fit_page = None
    kwargs = None
    settings = None
    _pop_up = None

    def run(self):
        try:
            gen_without_change = self.kwargs.pop('gen_without_change')
            load_file = self.kwargs.pop('load_file')
            if not load_file:
                fname1 = backup_folder / 'ga_fit.csv'
                if fname1.exists():
                    fname1.unlink()

            self.kwargs['fname'] = str(backup_folder / 'ga_fit.csv')
            evaluator = Evaluate(self.settings)
            ga = GAFit(evaluator.cf_matrix, evaluator.e_dft, **self.kwargs)
            best_indx = ga.run(gen_without_change=gen_without_change)

            cf_names = [evaluator.cf_names[i] for i, flag in enumerate(best_indx) if flag]
            optimizer = ECIOptimizer()
            optimizer.fit_page = self.fit_page
            optimizer.evaluator = Evaluate(self.settings,
                                           fitting_scheme=LinearRegression(),
                                           cf_names=cf_names)
            optimizer.optimize()
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)


class FitPage(Screen):
    graphs_added = False
    # Initialize standard fitting params
    fitting_params = {'algorithm': 'LASSO', 'alpha': 0.001}
    eci = {}
    fit_result = {}
    _pop_up = None
    fitting_in_progress = False
    e_dft = None
    e_ce = None

    energy_plot = None
    eci_plots = []
    energy_graph = None
    eci_graph = None
    zero_line_energy = None

    legend = None

    def on_enter(self):
        self.load_fit_alg_settings(self.fitting_params['algorithm'])
        if self.legend is None:
            self.legend = Legend(self.ids.legend)
        if not self.graphs_added:
            self.energy_graph = Graph(xlabel='DFT energy (eV/atom)',
                                      ylabel="E_CE - E_DFT (meV/atom)",
                                      x_ticks_minor=0,
                                      x_ticks_major=10,
                                      y_ticks_major=10,
                                      y_grid_label=True,
                                      x_grid_label=True,
                                      padding=5,
                                      xlog=False,
                                      ylog=False,
                                      xmin=0.0,
                                      ymin=0.0,
                                      precision='%.2f')
            self.energy_plot = ScatterPlot(color=FOREGROUND_TEXT_COLOR, point_size=3)
            self.zero_line_energy = LinePlot(line_width=2, color=FOREGROUND_TEXT_COLOR)
            self.energy_graph.add_plot(self.energy_plot)
            self.energy_graph.add_plot(self.zero_line_energy)
            self.ids.energyPlot.add_widget(self.energy_graph)

            self.eci_graph = Graph(ylabel='ECI (eV/atom)',
                                   xlabel='Clusters',
                                   x_ticks_minor=0,
                                   x_ticks_major=10,
                                   y_ticks_major=10,
                                   y_grid_label=True,
                                   x_grid_label=True,
                                   padding=5,
                                   xlog=False,
                                   ylog=False,
                                   xmin=-1.0,
                                   ymin=0.0,
                                   precision='%.2f')

            for i in range(8):
                color = ECI_GRAPH_COLORS[i % len(ECI_GRAPH_COLORS)]
                color = get_color_from_hex(color)
                plot = BarPlot(color=color, bar_width=4, bar_spacing=1)
                self.eci_plots.append(plot)
                self.eci_graph.add_plot(plot)

            self.ids.eciPlot.add_widget(self.eci_graph)
            self.graphs_added = True

    def dismiss_popup(self):
        if self._pop_up is None:
            return

        if isinstance(self._pop_up.content, FitAlgEditor):
            self._pop_up.content.backup()
        self._pop_up.dismiss()
        self._pop_up = None

    def show_lasso_editor(self):
        content = LassoEditor(close=self.close_lasso_editor)
        self._pop_up = Popup(title="LASSO Editor",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_l2_editor(self):
        content = L2Editor(close=self.close_l2_editor)
        self._pop_up = Popup(title="L2 Editor",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_bcs_editor(self):
        content = BCSEditor(close=self.close_bcs_editor)

        self._pop_up = Popup(title="BCS Editor",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_ga_editor(self):
        content = GAEditor(close=self.close_ga_editor)

        self._pop_up = Popup(title="Genetic Algorithm Editor",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def close_l2_editor(self, alpha):
        self.fitting_params = {}
        self.fitting_params['algorithm'] = 'L2'
        self.fitting_params['alpha'] = float(alpha)
        self.dismiss_popup()

    def close_lasso_editor(self, alpha):
        self.fitting_params = {}
        self.fitting_params['algorithm'] = 'LASSO'
        self.fitting_params['alpha'] = float(alpha)
        self.dismiss_popup()

    def close_bcs_editor(self, shape_var, rate_var, shape_lamb, var_opt_start, init_lamb,
                         lamb_opt_start, max_iter, noise):
        self.fitting_params = {
            'algorithm': 'BCS',
            'shape_var': float(shape_var),
            'rate_var': float(rate_var),
            'shape_lamb': float(shape_lamb),
            'var_opt_start': float(var_opt_start),
            'init_lamb': float(init_lamb),
            'lamb_opt_start': float(lamb_opt_start),
            'max_iter': int(max_iter),
            'noise': float(noise)
        }
        self.dismiss_popup()

    def close_ga_editor(self, elitism, mut_prob, num_individuals, max_active, cost_func, load_file,
                        max_without_imp):
        if isinstance(load_file, str):
            load_file = load_file == 'True'
        self.fitting_params = {
            'algorithm': 'GA',
            'elitism': int(elitism),
            'mut_prob': float(mut_prob),
            'num_individuals': int(num_individuals),
            'max_active': int(max_active),
            'cost_func': cost_func,
            'load_file': load_file,
            'gen_without_change': int(max_without_imp)
        }
        self.dismiss_popup()

    def show_load_ECI_file_dialog(self):
        content = LoadDialog(load=self.load_eci_file, cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load ECI file",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def load_eci_file(self, path, filename):
        if len(filename) == 0:
            fname = path
        else:
            fname = filename[0]

        self.ids.eciFileInput.text = fname
        self.dismiss_popup()

    def launch_fit_alg_editor(self):
        if self.ids.fitAlgSpinner.text == 'LASSO':
            self.show_lasso_editor()
        elif self.ids.fitAlgSpinner.text == 'L2':
            self.show_l2_editor()
        elif self.ids.fitAlgSpinner.text == 'BCS':
            self.show_bcs_editor()
        elif self.ids.fitAlgSpinner.text == 'Genetic Algorithm':
            self.show_ga_editor()

    def save_eci(self):
        fname = self.ids.eciFileInput.text

        if not fname.endswith('.json'):
            msg = 'ECI file should be a JSON file'
            App.get_running_app().root.ids.status.text = msg

        try:
            with open(fname, 'w') as out:
                json.dump(self.eci, out, separators=(',', ': '), indent=2)
            msg = f"ECIs saved to {fname}"
            App.get_running_app().root.ids.status.text = msg
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)

    def fit_eci(self):
        if self.fitting_in_progress:
            # We already optimising ECIs
            return

        settings = App.get_running_app().root.settings

        if settings is None:
            msg = 'Settings not set. Call Update settings first.'
            App.get_running_app().root.ids.status.text = msg
            return

        if self.ids.maxClusterDiaCut.text == '':
            max_cluster_dia_cut = None
        else:
            try:
                max_cluster_dia_cut = \
                    parse_max_cluster_dia(self.ids.maxClusterDiaCut.text)
            except Exception as exc:
                traceback.print_exc()
                App.get_running_app().root.ids.status.text = str(exc)
                return

        if self.ids.maxClusterSizeCut.text == '':
            max_cluster_size_cut = None
        else:
            max_cluster_size_cut = int(self.ids.maxClusterSizeCut.text)

        k_fold = self.ids.kFoldInput.text
        if k_fold == '':
            msg = 'K-fold has to be given'
            App.get_running_app().root.ids.status.text = msg
            return
        k_fold = int(k_fold)

        num_rep = self.ids.numRepititionsInput.text
        if num_rep == '':
            num_rep = 1
        else:
            num_rep = int(num_rep)

        scheme = self.fitting_params.get('algorithm', None)
        if scheme is None:
            msg = 'Open the fitting scheme editor prior to the fit'
            App.get_running_app().root.ids.status.text = msg
            return

        select_cond = self.ids.dbSelectCondInput.text
        if select_cond == '':
            select_cond = None
        else:
            select_cond = parse_select_cond(select_cond)

        scheme = scheme.lower()
        alpha = 0.0
        if scheme in ['lasso', 'l2']:
            alpha = self.fitting_params['alpha']
        elif scheme == 'bcs':
            from clease import BayesianCompressiveSensing
            scheme = BayesianCompressiveSensing(
                shape_var=self.fitting_params['shape_var'],
                rate_var=self.fitting_params['rate_var'],
                shape_lamb=self.fitting_params['shape_lamb'],
                variance_opt_start=self.fitting_params['var_opt_start'],
                lamb_opt_start=self.fitting_params['lamb_opt_start'],
                maxiter=self.fitting_params['max_iter'],
                noise=self.fitting_params['noise'],
                init_lamb=self.fitting_params['init_lamb'])
        elif scheme == 'ga':
            ga_runner = GAClusterSelector()
            ga_runner.fit_page = self
            gen_without_change = self.fitting_params['gen_without_change']
            ga_runner.kwargs = {
                'mutation_prob': self.fitting_params['mut_prob'],
                'elitism': self.fitting_params['elitism'],
                'fname': None,
                'num_individuals': self.fitting_params['num_individuals'],
                'max_num_in_init_pool': self.fitting_params['max_active'],
                'cost_func': self.fitting_params['cost_func'].lower(),
                'gen_without_change': gen_without_change,
                'load_file': self.fitting_params['load_file']
            }
            ga_runner.settings = settings

            # As GA behaves a bit different from the other schemes, we have
            # a separate runner and return after the runner is finished.
            msg = 'Selecting clusters with GA..'
            App.get_running_app().root.ids.status.text = msg
            Thread(target=ga_runner.run).start()
            return

        if k_fold == -1:
            scoring_scheme = 'loocv'
        else:
            scoring_scheme = 'k-fold'

        try:
            evaluator = Evaluate(settings,
                                 fitting_scheme=scheme,
                                 alpha=alpha,
                                 max_cluster_size=max_cluster_size_cut,
                                 max_cluster_dia=max_cluster_dia_cut,
                                 nsplits=k_fold,
                                 num_repetitions=num_rep,
                                 scoring_scheme=scoring_scheme,
                                 select_cond=select_cond)

            App.get_running_app().root.ids.status.text = 'Optimizing ECIs...'
            eci_optimizer = ECIOptimizer()
            eci_optimizer.fit_page = self
            eci_optimizer.evaluator = evaluator
            self.fitting_in_progress = True

            Thread(target=eci_optimizer.optimize).start()
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)

    def set_cv(self, cv):
        self.ids.cvLabel.text = f"CV: {cv:.3f} meV/atom"

    def set_rmse(self, rmse):
        self.ids.rmseLabel.text = f"RMSE: {rmse:.3f} meV/atom"

    def set_mae(self, mae):
        self.ids.maeLabel.text = f"MAE: {mae:.3f} meV/atom"

    def _eci_has_been_fitted(self):
        return self.e_dft is not None and self.e_ce is not None

    def update_energy_plot(self, e_dft, e_ce):
        if not self._eci_has_been_fitted():
            msg = 'ECIs has not been fitted yet'
            App.get_running_app().root.ids.status.text = msg
            return

        xmin = np.min(e_dft)
        xmax = np.max(e_dft)
        ymin = np.min(e_ce - e_dft) * 1000.0
        ymax = np.max(e_ce - e_dft) * 1000.0

        x_rng = xmax - xmin
        xmin -= 0.05 * x_rng
        xmax += 0.05 * x_rng

        y_rng = ymax - ymin
        ymin -= 0.05 * y_rng
        ymax += 0.05 * y_rng

        self.energy_graph.xmin = float(xmin)
        self.energy_graph.xmax = float(xmax)
        self.energy_graph.ymin = float(ymin)
        self.energy_graph.ymax = float(ymax)
        self.energy_graph.x_ticks_major = float(xmax - xmin) / 10.0
        self.energy_graph.y_ticks_major = float(ymax - ymin) / 10.0
        self.energy_graph.y_grid_label = True
        self.energy_graph.x_grid_label = True
        self.energy_plot.points = [(y, (x - y) * 1000.0) for x, y in zip(e_ce, e_dft)]
        self.zero_line_energy.points = [(xmin, 0.0), (xmax, 0.0)]

    def _clear_eci_plots(self):
        for plot in self.eci_plots:
            plot.points = []

    def update_eci_plot(self, eci):
        eci_by_size = {}

        if len(eci) == 0:
            return

        max_size = 0
        for k, v in eci.items():
            size = int((k[1]))
            if size < 2:
                continue
            if size not in eci_by_size.keys():
                eci_by_size[size] = []
            eci_by_size[size].append(v)

            if size > max_size:
                max_size = size

        sorted_keys = sorted(list(eci_by_size.keys()))
        prev = 0

        xmax = len(list(eci.keys())) + len(sorted_keys)
        ymin = min([v for _, v in eci.items()])
        ymax = max([v for _, v in eci.items()])
        self._clear_eci_plots()
        for i, k in enumerate(sorted_keys):
            values = eci_by_size[k]
            indx = range(prev, prev + len(values))
            self.eci_plots[i].points = list(zip(indx, values))
            prev = indx[-1] + 2

        y_rng = ymax - ymin
        ymin -= 0.05 * y_rng
        ymax += 0.05 * y_rng

        self.eci_graph.xmax = int(xmax)
        self.eci_graph.ymin = float(ymin)
        self.eci_graph.ymax = float(ymax)
        self.eci_graph.x_ticks_major = float(xmax) / 10.0
        self.eci_graph.y_ticks_major = float(ymax - ymin) / 10.0

        num_rows = (max_size - 2) // 3 + 1
        num_cols = min([max_size - 1, 3])

        legend_items = []
        for k in sorted_keys:
            color = get_color_from_hex(ECI_GRAPH_COLORS[int(k) - 2])[:3]
            legend_items.append({'text': f"{k}-body", 'color': color})
        self.legend.setup(legend_items, num_rows=num_rows, num_cols=num_cols)

    def update_plots(self):
        if self.fitting_in_progress:
            msg = 'Performing ECI opmisation in the background. '
            msg += 'Wait until the process is finished.'
            App.get_running_app().root.ids.status.text = msg

        self.update_energy_plot(self.e_dft, self.e_ce)
        self.update_eci_plot(self.eci)

    def to_dict(self):
        return {
            'eci_file': self.ids.eciFileInput.text,
            'db_select_cond': self.ids.dbSelectCondInput.text,
            'max_cluster_size': self.ids.maxClusterSizeCut.text,
            'max_cluster_dia': self.ids.maxClusterDiaCut.text,
            'k_fold': self.ids.kFoldInput.text,
            'num_repetitions': self.ids.numRepititionsInput.text,
            'fit_alg': self.ids.fitAlgSpinner.text
        }

    def from_dict(self, data):
        self.ids.eciFileInput.text = data.get('eci_file', '')
        self.ids.dbSelectCondInput.text = data.get('db_select_cond', '')
        self.ids.maxClusterSizeCut.text = data.get('max_cluster_size', '')
        self.ids.maxClusterDiaCut.text = data.get('max_cluster_dia', '')
        self.ids.kFoldInput.text = data.get('k_fold', '10')
        self.ids.numRepititionsInput.text = data.get('num_repetitions', '1')
        self.ids.fitAlgSpinner.text = data.get('fit_alg', 'LASSO')

    def load_fit_alg_settings(self, text):
        fnames = {
            'LASSO': LassoEditor.backup_file,
            'L2': L2Editor.backup_file,
            'BCS': BCSEditor.backup_file,
            'Genetic Algorithm': GAEditor.backup_file
        }

        closing_methods = {
            'LASSO': self.close_lasso_editor,
            'L2': self.close_l2_editor,
            'BCS': self.close_bcs_editor,
            'Genetic Algorithm': self.close_ga_editor
        }

        editors = {
            'LASSO': LassoEditor(),
            'L2': L2Editor(),
            'BCS': BCSEditor(),
            'Genetic Algorithm': GAEditor()
        }
        fname = fnames.get(text, None)
        close = closing_methods[text]

        app = App.get_running_app()
        if fname is None:
            app.root.ids.status.text = 'Unkown fitting scheme'
            return

        full_name = backup_folder / fname
        if not full_name.exists():
            # Create the file
            editors[text].backup()

        args = []
        with full_name.open('r') as fd:
            for line in fd:
                args.append(line.strip())
        args = args[::-1]
        try:
            close(*args)
            msg = 'Fit settings set the ones used '
            msg += 'last time'
            app.root.ids.status.text = msg
        except Exception:
            msg = 'Failed load previous fitting settings. '
            msg += 'Please set your settings again in the editor.'
            app.root.ids.status.text = msg
