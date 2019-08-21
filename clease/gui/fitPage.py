from kivy.uix.screenmanager import Screen
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivyagg")
from matplotlib import pyplot as plt
from kivy.uix.popup import Popup
from clease.gui.fittingAlgorithmEditors import LassoEditor, L2Editor, BCSEditor
from clease.gui.fittingAlgorithmEditors import GAEditor, FitAlgEditor
from clease.gui.load_save_dialog import LoadDialog
from kivy.app import App
import json
from clease.gui.util import parse_max_cluster_dia
from clease.gui.constants import BACKGROUND_COLOR, FOREGROUND_TEXT_COLOR
from threading import Thread


class ECIOptimiser(object):
    fit_page = None
    evaluator = None

    def optimise(self):
        try:
            self.fit_page.eci = self.evaluator.get_eci_dict()

            e_ce = self.evaluator.cf_matrix.dot(self.evaluator.eci)

            self.fit_page.e_ce = e_ce
            self.fit_page.e_dft = self.evaluator.e_dft
            self.fit_page.e_pred_leave_out = self.evaluator.e_pred_loo
            cv = self.evaluator.get_cv_score()
            rmse = self.evaluator.rmse()*1000.0
            mae = self.evaluator.mae()*1000.0
            self.fit_page.set_cv(cv)
            self.fit_page.set_rmse(rmse)
            self.fit_page.set_mae(mae)
            self.fit_page.ids.status.text = 'Idle'
            self.fit_page.fitting_in_progress = False
        except Exception as exc:
            self.fit_page.ids.status.text = str(exc)
            self.fit_page.fitting_in_progress = False
            return


class GAClusterSelector(object):
    fit_page = None
    kwargs = None
    settings = None
    _pop_up = None

    def run(self):
        from clease import GAFit, LinearRegression, Evaluate

        try:
            ga = GAFit(self.settings, **self.kwargs)
            _ = ga.run()

            optimiser = ECIOptimiser()
            optimiser.fit_page = self.fit_page

            max_cluster_dia = self.kwargs['max_cluster_dia']
            max_cluster_size = self.kwargs['max_cluster_size']
            select_cond = self.kwargs['select_cond']
            min_weight = self.kwargs['min_weight']
            optimiser.evaluator = Evaluate(
                self.settings, max_cluster_dia=max_cluster_dia,
                max_cluster_size=max_cluster_size,
                select_cond=select_cond, min_weight=min_weight,
                fitting_scheme=LinearRegression())
            optimiser.optimise()
        except Exception as exc:
            self.fit_page.ids.status.text = str(exc)


class FitPage(Screen):
    graphs_added = False
    fitting_params = {}
    eci = {}
    fit_result = {}
    _pop_up = None
    fitting_in_progress = False
    fit_on_separate_thread = True
    e_dft = None
    e_ce = None

    def on_enter(self):
        if not self.graphs_added:
            fig = plt.figure()
            fig.patch.set_facecolor(BACKGROUND_COLOR)
            ax = fig.add_subplot(1, 1, 1)
            ax.spines['bottom'].set_color(FOREGROUND_TEXT_COLOR)
            ax.spines['left'].set_color(FOREGROUND_TEXT_COLOR)
            ax.xaxis.label.set_color(FOREGROUND_TEXT_COLOR)
            ax.yaxis.label.set_color(FOREGROUND_TEXT_COLOR)
            ax.tick_params(axis='x', colors=FOREGROUND_TEXT_COLOR)
            ax.tick_params(axis='y', colors=FOREGROUND_TEXT_COLOR)
            ax.set_xlabel("DFT energy (eV/atom)")
            ax.set_ylabel("E_CE - E_DFT (meV/atom)")
            ax.set_facecolor(BACKGROUND_COLOR)
            self.ids.energyPlot.add_widget(FigureCanvasKivyAgg(fig))

            eci_fig = plt.figure()
            eci_fig.patch.set_facecolor(BACKGROUND_COLOR)
            ax = eci_fig.add_subplot(1, 1, 1)
            ax.spines['bottom'].set_color(FOREGROUND_TEXT_COLOR)
            ax.spines['left'].set_color(FOREGROUND_TEXT_COLOR)
            ax.xaxis.label.set_color(FOREGROUND_TEXT_COLOR)
            ax.yaxis.label.set_color(FOREGROUND_TEXT_COLOR)
            ax.tick_params(axis='x', colors=FOREGROUND_TEXT_COLOR)
            ax.tick_params(axis='y', colors=FOREGROUND_TEXT_COLOR)
            ax.set_ylabel("ECI (eV/atom)")
            ax.set_facecolor(BACKGROUND_COLOR)
            self.ids.eciPlot.add_widget(FigureCanvasKivyAgg(eci_fig))

            self.graphs_added = True

    def dismiss_popup(self):

        if isinstance(self._pop_up.content, FitAlgEditor):
            self._pop_up.content.backup()
        self._pop_up.dismiss()
        self._pop_up = None

    def show_lasso_editor(self):
        content = LassoEditor(close=self.close_lasso_editor)
        self._pop_up = Popup(title="LASSO Editor", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_l2_editor(self):
        content = L2Editor(close=self.close_l2_editor)
        self._pop_up = Popup(title="L2 Editor", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_bcs_editor(self):
        content = BCSEditor(close=self.close_bcs_editor)

        self._pop_up = Popup(title="BCS Editor", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_ga_editor(self):
        content = GAEditor(close=self.close_ga_editor)

        self._pop_up = Popup(title="Genetic Algorithm Editor", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
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

    def close_bcs_editor(self, shape_var, rate_var, shape_lamb, var_opt_start,
                         init_lamb, lamb_opt_start, max_iter, noise):
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

    def close_ga_editor(self, elitism, mut_prob, num_individuals, max_active,
                        cost_func, sparsity, sub_clust):
        self.fitting_params = {
            'algorithm': 'GA',
            'elitism': int(elitism),
            'mut_prob': float(mut_prob),
            'num_individuals': int(num_individuals),
            'max_active': int(max_active),
            'cost_func': cost_func,
            'sparsity': float(sparsity),
            'sub_clust': sub_clust == 'Yes'
        }
        self.dismiss_popup()

    def show_load_ECI_file_dialog(self):
        content = LoadDialog(load=self.load_eci_file,
                             cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load ECI file", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
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
            self.ids.status.text = 'ECI file should be a JSON file'

        try:
            with open(fname, 'w') as out:
                json.dump(self.eci, out, separators=(',', ': '), indent=2)
            self.ids.status.text = 'ECIs saved to {}'.format(fname)
        except Exception as exc:
            self.ids.status.text = str(exc)

    def fit_eci(self):
        if self.fitting_in_progress:
            # We already optimising ECIs
            return

        from clease import Evaluate
        settings = App.get_running_app().settings

        if settings is None:
            msg = 'Settings not set. Call Update settings first.'
            self.ids.status.text = msg
            return

        if self.ids.maxClusterDiaCut.text == '':
            max_cluster_dia_cut = None
        else:
            try:
                max_cluster_dia_cut = \
                    parse_max_cluster_dia(self.ids.maxClusterDiaCut.text)
            except Exception as exc:
                self.ids.status.text = str(exc)
                return

        if self.ids.maxClusterSizeCut.text == '':
            max_cluster_size_cut = None
        else:
            max_cluster_size_cut = int(self.ids.maxClusterSizeCut.text)

        k_fold = self.ids.kFoldInput.text
        if k_fold == '':
            self.ids.status.text = 'K-fold has to be given'
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
            self.ids.status.text = msg
            return

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
                init_lamb=self.fitting_params['init_lamb']
            )
        elif scheme == 'ga':
            ga_runner = GAClusterSelector()
            ga_runner.fit_page = self
            ga_runner.kwargs = {
                'max_cluster_size': max_cluster_size_cut,
                'max_cluster_dia': max_cluster_dia_cut,
                'select_cond': None,
                'min_weight': 1.0,
                'mutation_prob': self.fitting_params['mut_prob'],
                'elitism': self.fitting_params['elitism'],
                'num_individuals': self.fitting_params['num_individuals'],
                'max_num_in_init_pool': self.fitting_params['max_active'],
                'cost_func': self.fitting_params['cost_func'].lower(),
                'sparsity_slope': self.fitting_params['sparsity'],
                'include_subclusters': self.fitting_params['sub_clust']
            }
            ga_runner.settings = settings

            # GA behaves a bit different from the other schems
            # therefore we have a separate runner and return
            # after the runner is finished...
            if self.fit_on_separate_thread:
                self.ids.status.text = 'Selecting clusters with GA..'
                Thread(target=ga_runner.run).start()
                return
            else:
                ga_runner.run()
                return

        if k_fold == -1:
            scoring_scheme = 'loocv'
        else:
            scoring_scheme = 'k-fold'

        try:
            evaluator = Evaluate(
                settings, fitting_scheme=scheme, alpha=alpha,
                max_cluster_size=max_cluster_size_cut,
                max_cluster_dia=max_cluster_dia_cut, nsplits=k_fold,
                num_repetitions=num_rep, scoring_scheme=scoring_scheme)

            self.ids.status.text = 'Optimising ECIs...'
            eci_optimiser = ECIOptimiser()
            eci_optimiser.fit_page = self
            eci_optimiser.evaluator = evaluator
            self.fitting_in_progress = True

            if self.fit_on_separate_thread:
                Thread(target=eci_optimiser.optimise).start()
            else:
                eci_optimiser.optimise()
                self.update_energy_plot(self.e_dft, self.e_ce)
                self.update_eci_plot(self.eci)
        except Exception as exc:
            self.ids.status.text = str(exc)

    def set_cv(self, cv):
        self.ids.cvLabel.text = 'CV: {:.3f} meV/atom'.format(cv)

    def set_rmse(self, rmse):
        self.ids.rmseLabel.text = 'RMSE: {:.3f} meV/atom'.format(rmse)

    def set_mae(self, mae):
        self.ids.maeLabel.text = 'MAE: {:.3f} meV/atom'.format(mae)

    def _eci_has_been_fitted(self):
        return self.e_dft is not None and self.e_ce is not None

    def update_energy_plot(self, e_dft, e_ce):
        if not self._eci_has_been_fitted():
            self.ids.status.text = 'ECIs has not been fitted yet'
            return
        graph = self.ids.energyPlot.children[0]
        ax = graph.figure.axes[0]
        ax.clear()
        ax.set_xlabel("DFT energy (eV/atom)")
        ax.set_ylabel("E_CE - E_DFT (meV/atom)")
        diff = [(x - y)*1000.0 for x, y in zip(e_ce, e_dft)]
        ax.axhline(0.0, ls='--')
        ax.plot(e_dft, diff, 'o', mfc='none')
        graph.figure.canvas.draw()

    def update_eci_plot(self, eci):
        graph = self.ids.eciPlot.children[0]
        ax = graph.figure.axes[0]
        ax.clear()
        ax.set_ylabel("ECI (eV/atom)")
        ax.axhline(y=0.0, ls='--')
        eci_by_size = {}

        for k, v in eci.items():
            size = int((k[1]))
            if size < 2:
                continue
            if size not in eci_by_size.keys():
                eci_by_size[size] = []
            eci_by_size[size].append(v)

        sorted_keys = sorted(list(eci_by_size.keys()))
        prev = 0
        for k in sorted_keys:
            values = eci_by_size[k]
            indx = range(prev, prev + len(values))
            ax.bar(indx, values)
            prev = indx[-1]+2
        ax.set_xticklabels([])
        ax.set_xlim(-1, prev)
        graph.figure.canvas.draw()

    def thread_check_box_active(self, active):
        self.fit_on_separate_thread = not active

    def update_plots(self):
        if self.fitting_in_progress:
            msg = 'Performing ECI opmisation in the background. '
            msg += 'Wait until the process is finished.'
            self.ids.status.text = msg

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
