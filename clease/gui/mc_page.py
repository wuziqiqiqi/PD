from kivy.uix.screenmanager import Screen
from kivy_garden.graph import Graph, LinePlot
from kivy.uix.popup import Popup
from clease.gui.help_message_popup import HelpMessagePopup
from kivy.app import App
from clease.gui.util import parse_concentration_list, parse_temperature_list
from clease.gui.constants import FOREGROUND_TEXT_COLOR
from clease.gui.mc_runner import MCRunner
from threading import Thread
import json
import os


class MCPage(Screen):
    energy_graph = None
    energy_plot = None
    _pop_up = None
    mc_is_running = False

    def on_enter(self):
        if self.energy_graph is None:
            self.energy_graph = Graph(
                xlabel='MC sweep',
                ylabel="E - E[0]",
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

            self.energy_plot = LinePlot(line_width=2,
                                        color=FOREGROUND_TEXT_COLOR)

            self.energy_graph.add_plot(self.energy_plot)
            self.ids.energyPlot.add_widget(self.energy_graph)

    def dismiss_popup(self):
        self._pop_up.dismiss()
        self._pop_up = None

    def show_temp_help_message(self):
        msg = 'Temperatures can be specified as a comma separated\n'
        msg += 'list of values. The temperatures are given in Kelvin\n'
        msg += 'Example:\n'
        msg += '1000, 900, 800, 700, 600, 500, 400, 300'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Temperature input", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_conc_help_message(self):
        msg = 'Concentratrions are given as a comma separated list per basis\n'
        msg += 'Example:\n\n'
        msg += 'If basis elements is (Au, Cu, X), (Cu, X)\n'
        msg += 'We can specify concentrations by\n'
        msg += '(0.5, 0.3, 0.2), (0.9, 0.1)\n'
        msg += 'It is important the the concentration sums to one in each '
        msg += 'basis\n'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Concentration input", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_size_help_message(self):
        msg = 'Scaling factor used to scale the currently active template\n'
        msg += 'If this factor is 5 a supercell consiting (5, 5, 5)\n'
        msg += 'extension of the active template will be used for the\n'
        msg += 'MC calculation\n'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Concentration input", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def to_dict(self):
        return {
            'temps': self.ids.tempInput.text,
            'concs': self.ids.concInput.text,
            'size': self.ids.sizeInput.text,
            'sweeps': self.ids.sweepInput.text
        }

    def from_dict(self, dct):
        self.ids.tempInput.text = dct.get('temps', '')
        self.ids.concInput.text = dct.get('concs', '')
        self.ids.sizeInput.text = dct.get('size', '1')
        self.ids.sweepInput.text = dct.get('sweeps', '100')

    def runMC(self):
        if self.mc_is_running:
            return

        try:
            temps = parse_temperature_list(self.ids.tempInput.text)
            concs = parse_concentration_list(self.ids.concInput.text)
            size = int(self.ids.sizeInput.text)

            app = App.get_running_app()
            settings = app.root.settings

            if settings is None:
                app.root.ids.status.text = 'Apply settings prior to running MC'
                return

            eci_file = app.root.ids.sm.get_screen('Fit').ids.eciFileInput.text
            if not os.path.exists(eci_file):
                msg = 'Cannot load ECI from {}. No such file.'.format(eci_file)
                app.root.ids.status.text = msg
                return

            with open(eci_file, 'r') as infile:
                eci = json.load(infile)

            atoms = settings.atoms*(size, size, size)
            sweeps = int(self.ids.sweepInput.text)

            runner = MCRunner(atoms=atoms, eci=eci, mc_page=self, conc=concs,
                              temps=temps, settings=settings, sweeps=sweeps,
                              status=app.root.ids.status)

            Thread(target=runner.run).start()
        except Exception as exc:
            App.get_running_app().root.ids.status.text = str(exc)
