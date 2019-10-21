from kivy.uix.screenmanager import Screen
from kivy_garden.graph import Graph, LinePlot
from kivy.uix.popup import Popup
from clease.gui.help_message_popup import HelpMessagePopup
from kivy.app import App
from clease.gui.util import parse_concentration_list, parse_temperature_list
from clease.gui.util import parse_comma_sep_list_of_int
from clease.gui.constants import MC_MEAN_CURVE_COLOR
from clease.gui.constants import CONC_PER_BASIS, SYSTEMS_FROM_DB
from clease.gui.mc_runner import MCRunner
from kivy.utils import get_color_from_hex
from threading import Thread
from clease.gui.load_save_dialog import LoadDialog
import json
import os
import traceback


class MCPage(Screen):
    energy_graph = None
    energy_plot = None
    mean_energy_plot = None
    _pop_up = None
    mc_is_running = False
    active_template_is_mc_cell = False
    info_update_disabled = False
    view_mc_cell_disabled = False
    _mc = None

    def on_enter(self):
        self.set_cell_info()
        if self.energy_graph is None:
            self.energy_graph = Graph(
                xlabel='MC sweep',
                ylabel="Mean energy (<E> - <E>[0])",
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

            color = get_color_from_hex(MC_MEAN_CURVE_COLOR)
            self.mean_energy_plot = LinePlot(line_width=2, color=color)

            self.energy_graph.add_plot(self.mean_energy_plot)
            self.ids.energyPlot.add_widget(self.energy_graph)

    def bind_mc(self, mc):
        self._mc = mc

    def detach_mc(self):
        self._mc = None

    def abort_mc(self):
        if self._mc is not None:
            self._mc.quit = True

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
        msg += 'basis\n\n'
        msg += 'Another option is to give a comma separated list of ids.\n'
        msg += 'The numbers are then interpreted as IDs in the MC database.\n'
        msg += 'To run a sequence of MC calculations, you can therefore \n'
        msg += 'prepare a set of structures and store in the DB.\n'
        msg += 'It is important that the structures match the MC cell.\n'
        msg += 'The easiest way is to click on View MC cell and store it as \n'
        msg += 'xyz file and use that as a template for your simulation\n'
        msg += 'cells.'
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
            'sweeps': self.ids.sweepInput.text,
            'mc_db': self.ids.mc_db_input.text
        }

    def from_dict(self, dct):
        self.ids.tempInput.text = dct.get('temps', '')
        self.ids.concInput.text = dct.get('concs', '')
        self.ids.sizeInput.text = dct.get('size', '1')
        self.ids.sweepInput.text = dct.get('sweeps', '100')
        self.ids.mc_db_input.text = dct.get('mc_db', '')

    def view_mc_cell(self):
        app = App.get_running_app()
        if self.view_mc_cell_disabled:
            msg = 'Cannot view MC cell while attaching calculator'
            app.root.ids.status.text = msg
            return
        try:
            from ase.visualize import view
            atoms = self._get_mc_cell()
            Thread(target=view, args=(atoms,)).start()
        except Exception as exc:
            traceback.print_exc()
            app.root.ids.status.text = str(exc)

    def _get_mc_cell(self):
        app = App.get_running_app()
        try:
            settings = app.root.settings
            size = int(self.ids.sizeInput.text)

            if settings is None:
                app.root.ids.status.text = 'Apply settings prior to running MC'
                return

            atoms = None
            if self.active_template_is_mc_cell:
                atoms = settings.atoms.copy()
            else:
                atoms = settings.atoms*(size, size, size)
            return atoms
        except Exception as exc:
            traceback.print_exc()
            app.root.ids.status.text = str(exc)
        return None

    def set_cell_info(self):
        if self.info_update_disabled:
            return
        atoms = self._get_mc_cell()

        if atoms is None:
            return

        info = atoms.get_cell_lengths_and_angles()

        length_str = 'a: {}Å b: {}Å c: {}Å'.format(
            int(info[0]), int(info[1]), int(info[2]))
        self.ids.mc_cell_lengths.text = length_str

        angle_str = '{}deg {}deg {}deg'.format(int(info[3]), int(info[4]),
                                               int(info[5]))
        self.ids.mc_cell_angles.text = angle_str
        self.ids.mc_num_atoms.text = 'Num atoms: {}'.format(len(atoms))

    def open_load_dialog(self):
        content = LoadDialog(load=self.load_mc_db_file,
                             cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load MC DB file", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def load_mc_db_file(self, path, filename):
        if len(filename) == 0:
            fname = path
        else:
            fname = filename[0]

        self.ids.mc_db_input.text = fname
        self.dismiss_popup()

    def show_db_help_message(self):
        msg = 'ASE database used to store thermodynamic data from MC runs\n'
        msg += 'The data are stored in an external table called thermo_data.\n'
        msg += 'The data can thus be accessed from an AtomsRow object by\n'
        msg += "row['thermo_data']"
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="MC DB", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def runMC(self):
        if self.mc_is_running:
            return

        try:
            temps = parse_temperature_list(self.ids.tempInput.text)
            conc_mode = None

            # First try to parse as list
            conc_string = self.ids.concInput.text
            try:
                concs = parse_comma_sep_list_of_int(conc_string)
                conc_mode = SYSTEMS_FROM_DB
            except:
                pass

            if conc_mode is None:
                try:
                    concs = parse_concentration_list(conc_string)
                    conc_mode = CONC_PER_BASIS
                except:
                    pass

            if conc_mode is None:
                msg = 'Cannot parse concentration'
                App.get_running_app().root.ids.status.text = msg
                return

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
            db_name = self.ids.mc_db_input.text

            runner_args = dict(atoms=atoms, eci=eci, mc_page=self, conc=concs,
                               temps=temps, settings=settings, sweeps=sweeps,
                               status=app.root.ids.status, db_name=db_name,
                               conc_mode=conc_mode)
            if conc_mode == CONC_PER_BASIS:
                runner = MCRunner(**runner_args)
            else:
                # Create a chain of MCRunners
                runner_args['conc'] = None
                runner_args['db_id'] = concs[0]
                runner = MCRunner(**runner_args)
                current_runner = runner
                for i in range(1, len(concs)):
                    runner_args['db_id'] = concs[1]
                    current_runner.next_mc_obj = MCRunner(**runner_args)
                    current_runner = current_runner.next_mc_obj

            Thread(target=runner.run).start()
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
