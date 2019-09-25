from kivy.uix.screenmanager import Screen
from kivy_garden.graph import Graph, LinePlot
from kivy.uix.popup import Popup
from clease.gui.help_message_popup import HelpMessagePopup
from clease.gui.constants import FOREGROUND_TEXT_COLOR


class MCPage(Screen):
    energy_graph = None
    energy_plot = None
    _pop_up = None

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

        
    def runMC(self):
        pass
