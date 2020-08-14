from kivy.uix.screenmanager import Screen
from kivy_garden.graph import Graph, LinePlot
from clease.gui.constants import (FOREGROUND_TEXT_COLOR, META_DYN_MSG)
from kivy.app import App
from kivy.uix.popup import Popup
from clease.gui.help_message_popup import HelpMessagePopup
from clease.gui.conc_obs_editor import ConcObsEditor
from clease.gui.sgc_editor import SGCEditor
from threading import Thread
from clease.gui.meta_dyn_runner import MetaDynRunner
from clease.montecarlo import BinnedBiasPotential
from clease.montecarlo.observers import ConcentrationObserver
import json
import os
import numpy as np
from clease.tools import species_chempot2eci


class MetaDynPage(Screen):
    _pop_up = None
    observer_params = {}
    ensemble_params = {}
    visit_graph = None
    visit_plot = None
    visit_mean_plot = None
    visit_conv_plot = None
    beta_free_energy_graph = None
    beta_free_energy_plot = None
    mc_is_running = False
    _meta_dyn_sampler = None

    def on_enter(self):
        app = App.get_running_app()
        app.root.mc_type_screen = 'MetaDynPage'

        try:
            if self.observer_params == {}:
                # Open and close the observer editor
                self.launch_observer_editor(open_popup=False)
                self._pop_up.content.ids.closeButton.dispatch('on_release')
            if self.ensemble_params == {}:
                self.launch_ensemble_editor(open_popup=False)
                self._pop_up.content.ids.closeButton.dispatch('on_release')
        except Exception:
            pass

        if self.beta_free_energy_graph is None:
            self.beta_free_energy_graph = Graph(xlabel='Collective variable',
                                                ylabel="G/kT",
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
                                                precision='%1.1e')

            self.beta_free_energy_plot = LinePlot(line_width=1, color=FOREGROUND_TEXT_COLOR)
            self.beta_free_energy_graph.add_plot(self.beta_free_energy_plot)
            self.ids.freeEnergyPlot.add_widget(self.beta_free_energy_graph)

            self.visit_graph = Graph(xlabel='Collective variable',
                                     ylabel="Num. visits",
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
                                     precision='%1.1e')
            self.visit_plot = LinePlot(line_width=1, color=FOREGROUND_TEXT_COLOR)
            self.visit_mean_plot = LinePlot(line_width=1, color=FOREGROUND_TEXT_COLOR)
            self.visit_conv_plot = LinePlot(line_width=1, color=FOREGROUND_TEXT_COLOR)
            self.visit_graph.add_plot(self.visit_plot)
            self.visit_graph.add_plot(self.visit_mean_plot)
            self.visit_graph.add_plot(self.visit_conv_plot)
            self.ids.visitHistogram.add_widget(self.visit_graph)

    def bind_meta_dyn_sampler(self, sampler):
        self._meta_dyn_sampler = sampler

    def detach_meta_dyn_sampler(self):
        self._meta_dyn_sampler = None

    def abort_mc(self):
        if self._meta_dyn_sampler is not None:
            self._meta_dyn_sampler.quit = True
            app = App.get_running_app()
            app.root.ids.status.text = META_DYN_MSG['abort_mc']

    def dismiss_popup(self):
        if self._pop_up is None:
            return
        self._pop_up.dismiss()
        self._pop_up = None

    def show_temperature_help(self):
        msg = 'Temperature used during sampling given in Kelvin'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Temperature input",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_var_help(self):
        msg = 'Collective variable used to map out the \n'
        msg += 'free energy.'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Collective variable",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_range_help(self):
        msg = 'The simulation will be limited to configurations\n'
        msg += 'where the collective variable is within the \n'
        msg += 'specified range.'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Range help",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_flatness_help(self):
        msg = 'The histogram of visits is considered to be\n'
        msg += 'flat, when the bin with the minimum value\n'
        msg += 'is larger that <flatness criteria> times the\n'
        msg += 'mean.\n'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Flatness help",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_nbins_help(self):
        msg = 'Number of intervals which is used to partition\n'
        msg += 'the interval.\n'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Num. bins help",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_modfactor_help(self):
        msg = 'The modification factor is a penalisation value \n'
        msg += 'added to the free energy curve when the sampler\n'
        msg += 'visits a particular bin. It is given in units of\n'
        msg += 'kT. Hence, if it is 0.01, it means that the energy\n'
        msg += 'is penalised by 0.01*kT.'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Mod. factor help",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_max_sweep_help(self):
        msg = 'Maximum number of sweeps before the calculation\n'
        msg += 'will be terminated. If it converges before reaching\n'
        msg += 'this number, it stops earlier\n'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Max. sweeps help",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def close_conc_editor(self, element):
        self._pop_up.content.backup()
        self.observer_params = {'name': 'Concentration', 'element': element}
        self.dismiss_popup()

    def launch_observer_editor(self, var=None, open_popup=True):
        if var is None:
            var = self.ids.varSpinner.text

        content = None
        if var == 'Concentration':
            default_elem = sorted(self.unique_symbols())[0]
            content = ConcObsEditor(close=self.close_conc_editor, default_element=default_elem)

        if content is None:
            return
        self._pop_up = Popup(title=var,
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        if open_popup:
            self._pop_up.open()

    def launch_ensemble_editor(self, text=None, open_popup=True):
        if text is None:
            text = self.ids.ensembleSpinner.text

        content = None
        if text == 'Semi-grand canonical':
            symbols = self.unique_symbols()
            default_symbs = ', '.join(symbols)
            def_chem_pot = ', '.join([f"{s}: 0.0" for s in symbols[:-1]])
            content = SGCEditor(close=self.close_sgc_editor,
                                symbols=default_symbs,
                                chem_pot=def_chem_pot)

        if content is None:
            return

        self._pop_up = Popup(title=text,
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        if open_popup:
            self._pop_up.open()

    def close_sgc_editor(self, symbols, chem_pot):
        self._pop_up.content.backup()
        self.ensemble_params = {
            'name': 'Semi-grand canonical',
            'symbols': symbols,
            'chem_pot': chem_pot
        }
        self.dismiss_popup()

    def run(self):
        """
        Run the meta dynamics sampler
        """
        if self.mc_is_running:
            return
        app = App.get_running_app()
        settings = app.root.settings
        if settings is None:
            app.root.ids.status.text = META_DYN_MSG['settings_is_none']
            return

        if len(settings.concentration.basis_elements) > 1:
            app.root.ids.status.text = META_DYN_MSG['more_than_one_basis']
            return

        # Load ECI
        mc_header = app.root.ids.sm.get_screen('MCHeader')
        main_mc_page = mc_header.ids.sm.get_screen('MCMainPage')
        eci_file = main_mc_page.ids.eciFileInput.text

        if not os.path.exists(eci_file):
            msg = META_DYN_MSG['no_eci'] + f"{eci_file}"
            app.root.ids.status.text = msg
            return

        with open(eci_file, 'r') as infile:
            eci = json.load(infile)

        try:
            size = int(main_mc_page.ids.sizeInput.text)
        except Exception as exc:
            app.root.ids.status.text = str(exc)
            return

        if app.root.active_template_is_mc_cell:
            app.root.ids.status.text = META_DYN_MSG['mc_cell_is_template']
            return

        atoms = settings.atoms * (size, size, size)

        ensemble = self.ids.ensembleSpinner.text

        try:
            T = float(self.ids.tempInput.text)
            xmin = float(self.ids.varMinInput.text)
            xmax = float(self.ids.varMaxInput.text)
            nbins = int(self.ids.nbinsInput.text)
            max_sweeps = int(self.ids.maxSweepsInput.text)
            flat_limit = float(self.ids.flatInput.text)
            mod_factor = float(self.ids.modInput.text)
            update_int = int(self.ids.plotIntInput.text)
        except Exception as exc:
            app.root.ids.status.text = str(exc)
            return

        getter = None
        variable = self.ids.varSpinner.text
        if variable == 'Concentration':
            if 'element' not in self.observer_params.keys():
                msg = META_DYN_MSG['var_editor_not_launched']
                app.root.ids.status.text = msg
                return
            elem = self.observer_params['element']
            getter = ConcentrationObserver(atoms, element=elem)
        else:
            msg = META_DYN_MSG['unkown_var'] + variable
            app.root.ids.status.text = msg
            return

        pot = BinnedBiasPotential(xmin=xmin, xmax=xmax, nbins=nbins, getter=getter)

        fname = self.ids.backupInput.text + '.json'
        if os.path.exists(fname):
            with open(fname, 'r') as infile:
                data = json.load(infile)
            pot.from_dict(data['bias_pot'])

        if ensemble == 'Semi-grand canonical':
            if 'symbols' not in self.ensemble_params.keys():
                app.root.ids.status.text = META_DYN_MSG['launch_ens_editor']
                return
            chem_pot = self.ensemble_params['chem_pot']
            bf_list = settings.basis_functions
            chem_pot_eci = species_chempot2eci(bf_list, chem_pot)
            for k, v in chem_pot_eci.items():
                eci[k] = eci.get(k, 0.0) - v
        else:
            app.root.ids.status.text = META_DYN_MSG['unknown_ens'] + ensemble
            return

        runner = MetaDynRunner(atoms=atoms,
                               meta_page=self,
                               max_sweeps=max_sweeps,
                               settings=settings,
                               app_root=app.root,
                               eci=eci,
                               status=app.root.ids.status,
                               mc_params=self.ensemble_params,
                               T=T,
                               bias=pot,
                               flat=flat_limit,
                               mod_factor=mod_factor,
                               backup_file=fname,
                               update_interval=update_int)
        Thread(target=runner.run).start()

    def to_dict(self):
        return {
            'temperature': self.ids.tempInput.text,
            'collective_var': self.ids.varSpinner.text,
            'min_collective_var': self.ids.varMinInput.text,
            'max_collective_var': self.ids.varMaxInput.text,
            'flatness': self.ids.flatInput.text,
            'nbins': self.ids.nbinsInput.text,
            'backup': self.ids.backupInput.text,
            'ensemble': self.ids.ensembleSpinner.text,
            'mod_factor': self.ids.modInput.text,
            'max_sweeps': self.ids.maxSweepsInput.text,
            'plot_update_interval': self.ids.plotIntInput.text
        }

    def from_dict(self, dct):
        """
        Initialise the fields from a dictionary
        """
        self.ids.tempInput.text = dct.get('temperature', '500')
        self.ids.varSpinner.text = dct.get('collective_var', 'Concentration')
        self.ids.varMinInput.text = dct.get('min_collective_var', '0.0')
        self.ids.varMaxInput.text = dct.get('max_collective_var', '1.0')
        self.ids.flatInput.text = dct.get('flatness', '0.8')
        self.ids.nbinsInput.text = dct.get('nbins', '50')
        self.ids.backupInput.text = dct.get('backup', 'metadyn')
        self.ids.ensembleSpinner.text = dct.get('ensemble', 'Semi-grand canonical')
        self.ids.modInput.text = dct.get('mod_factor', '0.1')
        self.ids.maxSweepsInput.text = dct.get('max_sweep', '10000')
        self.ids.plotIntInput.text = dct.get('plot_update_interval', '10')

    def update_free_energy_plot(self, x, betaG):
        """
        Updates the free energy plot

        Parameters:

        x: array
            Collective variable

        betaG: array
            Array with the free energy (divided by kT)
        """
        xmin = np.min(x)
        xmax = np.max(x)
        x_rng = xmax - xmin
        xmin -= 0.05 * x_rng
        xmax += 0.05 * x_rng

        betaG -= betaG[0]
        ymin = np.min(betaG)
        ymax = np.max(betaG)
        y_rng = ymax - ymin
        ymin -= 0.05 * y_rng
        ymax += 0.05 * y_rng

        self.beta_free_energy_graph.xmin = float(xmin)
        self.beta_free_energy_graph.xmax = float(xmax)
        self.beta_free_energy_graph.ymin = float(ymin)
        self.beta_free_energy_graph.ymax = float(ymax)
        self.beta_free_energy_graph.x_ticks_major = float(xmax - xmin) / 10.0
        self.beta_free_energy_graph.y_ticks_major = float(ymax - ymin) / 10.0
        self.beta_free_energy_graph.y_grid_label = True
        self.beta_free_energy_graph.x_grid_label = True
        self.beta_free_energy_plot.points = list(zip(x, betaG))

    def update_visit_plot(self, x, visits, flat_criteria):
        """
        Update the visit graph

        Parameters:

        x: array
            Array with the collective variables

        visits: array
            Array with the visit histogram

        flat_criteria: float
            Criteria used to judge when a histogram is flat
        """
        xmin = np.min(x)
        xmax = np.max(x)
        x_rng = xmax - xmin
        xmin -= 0.05 * x_rng
        xmax += 0.05 * x_rng

        ymin = np.min(visits)
        ymax = np.max(visits)
        y_rng = ymax - ymin
        ymin -= 0.05 * y_rng
        ymax += 0.05 * y_rng

        self.visit_graph.xmin = float(xmin)
        self.visit_graph.xmax = float(xmax)
        self.visit_graph.ymin = float(ymin)
        self.visit_graph.ymax = float(ymax)
        self.visit_graph.x_ticks_major = float(xmax - xmin) / 10.0
        self.visit_graph.y_ticks_major = float(ymax - ymin) / 10.0
        self.visit_graph.y_grid_label = True
        self.visit_graph.x_grid_label = True
        self.visit_plot.points = list(zip(x, visits))
        mean = np.mean(visits)
        self.visit_mean_plot.points = [(xmin, mean), (xmax, mean)]
        conv = flat_criteria * mean
        self.visit_conv_plot.points = [(xmin, conv), (xmax, conv)]

    def clear_backup(self):
        fname = self.ids.backupInput.text + '.json'

        if os.path.exists(fname):
            os.remove(fname)

    def unique_symbols(self):
        app = App.get_running_app()
        settings = app.root.settings

        if settings is None:
            app.root.ids.status.text = 'Settings is not set'
            return ['']

        unique_symb = settings.unique_element_without_background()
        return unique_symb
