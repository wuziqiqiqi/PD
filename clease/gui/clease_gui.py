from kivy.app import App
from kivy.lang import Builder
from kivy.uix.stacklayout import StackLayout
from kivy.resources import resource_add_path
from kivy.uix.popup import Popup

from clease.gui.settings_page import SettingsPage
from clease.gui.concentration_page import ConcentrationPage
from clease.gui.new_struct_page import NewStructPage
from clease.gui.fit_page import FitPage
from clease.gui.meta_dyn_page import MetaDynPage
from clease.gui.reconfig_db import ReconfigDB
from kivy.core.window import Window
from clease.gui.job_exec import JobExec
from clease.gui.mc_header import MCHeader
from clease.gui.load_save_dialog import LoadDialog, SaveDialog
from clease.gui.db_browser import DbBrowser
from clease import Evaluate
import signal
from pathlib import Path
import traceback

try:
    import ase.db.app as ase_db_webapp
    from flask import request
    has_flask = True
except ImportError:
    has_flask = False

import json
from threading import Thread

import os

main_path = Path(__file__)
resource_add_path(main_path.parent / 'layout')

Builder.load_file("clease_gui.kv")


class WindowFrame(StackLayout):
    _pop_up = None
    current_session_file = None
    settings = None
    reconfig_in_progress = False
    subprocesses = {}
    active_template_is_mc_cell = False
    info_update_disabled = False
    view_mc_cell_disabled = False

    def dismiss_popup(self):
        if self._pop_up is None:
            return
        self._pop_up.dismiss()
        self._pop_up = None

    def show_load_session_dialog(self):
        content = LoadDialog(load=self.load_session, cancel=self.dismiss_popup)
        self._pop_up = Popup(title="Load CLEASE session",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def load_session(self, path, filename):
        try:
            with open(filename[0], 'r') as infile:
                data = json.load(infile)

            # variables for "Concentration" screen
            conc_page = self.ids.sm.get_screen('Concentration')
            elements = data['conc']['elements']
            grouped_basis = data['conc']['grouped_basis']
            conc_page.set_Elements_GroupedBasis(elements, grouped_basis)

            A_lb = data['conc']['A_lb']
            rhs_lb = data['conc']['rhs_lb']
            A_eq = data['conc']['A_eq']
            rhs_eq = data['conc']['rhs_eq']
            conc_page.load_from_matrices(A_lb, rhs_lb, A_eq, rhs_eq)

            # variables for "Settings" screen
            settings_page = self.ids.sm.get_screen('Settings')
            settings_page.from_dict(data.get('settings', {}))
            settings_page.apply_settings()

            newstruct_page = self.ids.sm.get_screen('NewStruct')
            newstruct_page.from_dict(data.get('new_struct', {}))

            job_exec = self.ids.sm.get_screen('JobExec')
            job_exec.from_dict(data.get('job_exec', {}))

            fit_page = self.ids.sm.get_screen('Fit')
            fit_page.from_dict(data.get('fit_page', {}))

            mc_header = self.ids.sm.get_screen('MCHeader')
            mc_main_page = mc_header.ids.sm.get_screen('MCMainPage')
            mc_main_page.from_dict(data.get('mc_main', {}))
            canonical_page = mc_header.ids.sm.get_screen('MC')
            canonical_page.from_dict(data.get('canonical_mc', {}))

            meta_dyn_page = mc_header.ids.sm.get_screen('MetaDynPage')
            meta_dyn_page.from_dict(data.get('meta_dyn_page', {}))

            self.current_session_file = filename[0]
        except Exception as e:
            msg = "An error occured during load: " + str(e)
            App.get_running_app().root.ids.status.text = msg

        self.dismiss_popup()

    def save_session_to_current_file(self):
        if self.current_session_file is not None:
            self.save_session(None, [self.current_session_file], None)

        else:
            self.show_save_dialog()

    def save_session(self, path, selection, user_filename):
        if self.ids.sm.get_screen('Settings').check_user_input() != 0:
            return

        if self.ids.sm.get_screen('Concentration').check_user_input() != 0:
            return

        if len(selection) == 0:
            fname = str(Path(path) / user_filename)
        else:
            fname = selection[0]

        data = {}
        data['settings'] = self.ids.sm.get_screen('Settings').to_dict()
        data['conc'] = self.ids.sm.get_screen('Concentration').to_dict()
        data['new_struct'] = self.ids.sm.get_screen('NewStruct').to_dict()
        data['job_exec'] = self.ids.sm.get_screen('JobExec').to_dict()
        data['fit_page'] = self.ids.sm.get_screen('Fit').to_dict()
        mc_header = self.ids.sm.get_screen('MCHeader')
        data['mc_main'] = mc_header.ids.sm.get_screen('MCMainPage').to_dict()
        data['canonical_mc'] = mc_header.ids.sm.get_screen('MC').to_dict()
        data['meta_dyn_page'] = mc_header.ids.sm.get_screen('MetaDynPage').to_dict()

        with open(fname, 'w') as outfile:
            json.dump(data, outfile, separators=(',', ': '), indent=2)
        msg = f"Session saved to {fname}"
        App.get_running_app().root.ids.status.text = msg
        self.dismiss_popup()
        self.current_session_file = fname

    def show_save_dialog(self):
        if self.ids.sm.get_screen('Settings').check_user_input() != 0:
            return

        if self.ids.sm.get_screen('Concentration').check_user_input() != 0:
            return

        content = SaveDialog(save=self.save_session,
                             cancel=self.dismiss_popup,
                             fname="cleaseDemo.json")

        self._pop_up = Popup(title="Save CLEASE session",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def change_screen(self, new_screen):
        current = self.ids.sm.current
        all_screens = self.ids.sm.screen_names

        index_current = all_screens.index(current)
        index_new = all_screens.index(new_screen)

        direction = 'left'

        if index_current > index_new:
            direction = 'right'
        self.ids.sm.transition.direction = direction

        # Take special care of MC screens as
        if new_screen == 'MC':
            new_screen = self.mc_type_screen
        self.ids.sm.current = new_screen

    def reconfig(self):
        """Reconfigure DB entries based on current settings."""
        if self.reconfig_in_progress:
            # Do no allow user to initialize many threads
            return

        self.reconfig_in_progress = True

        reconfig = ReconfigDB()
        reconfig.app = App.get_running_app()
        reconfig.status = App.get_running_app().root.ids.status

        Thread(target=reconfig.reconfig_db).start()

    def view_clusters(self):
        """View clusters."""
        images = self._get_clusters()

        if images is False:
            msg = "Settings should be applied/loaded before viewing clusters."
            App.get_running_app().root.ids.status.text = str(msg)
            return

        try:
            from ase.visualize import view
            Thread(target=view, args=(images,)).start()

        except Exception as exc:
            App.get_running_app().root.ids.status.text = str(exc)

    def _get_clusters(self):
        if self.settings is None:
            self._apply_settings()

        try:
            return self.settings.cluster_mng.get_figures()
        except Exception:
            return False

    def _apply_settings(self):
        self.ids.sm.get_screen('Settings').apply_settings()

    def view_training_db(self):
        if self.settings is None:
            self._apply_settings()

        app = App.get_running_app()
        if self.settings is None:
            msg = 'Could not apply settings. Check your input.'
            app.root.ids.status.text = msg
            return

        screen = self.ids.sm.get_screen('Settings')
        db_name = screen.ids.dbNameInput.text

        content = DbBrowser(close=self.dismiss_popup, db_name=db_name)
        self._pop_up = Popup(title="DB Browser",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def get_mc_cell(self):
        app = App.get_running_app()
        try:
            settings = app.root.settings

            mc_header = self.ids.sm.get_screen('MCHeader')

            mc_main_page = mc_header.ids.sm.get_screen('MCMainPage')

            size = int(mc_main_page.ids.sizeInput.text)

            if settings is None:
                app.root.ids.status.text = 'Apply settings prior to running MC'
                return

            atoms = None
            if self.active_template_is_mc_cell:
                atoms = settings.atoms.copy()
            else:
                atoms = settings.prim_cell * (size, size, size)
            return atoms
        except Exception as exc:
            traceback.print_exc()
            app.root.ids.status.text = str(exc)
        return None

    def view_mc_cell(self):
        app = App.get_running_app()
        if self.view_mc_cell_disabled:
            msg = 'Cannot view MC cell while attaching calculator'
            app.root.ids.status.text = msg
            return
        try:
            from ase.visualize import view
            atoms = self.get_mc_cell()
            Thread(target=view, args=(atoms,)).start()
        except Exception as exc:
            traceback.print_exc()
            app.root.ids.status.text = str(exc)

    def export_fit_data(self, path, selection, user_filename):
        if len(selection) == 0:
            fname = str(Path(path) / user_filename)
        else:
            fname = selection[0]

        app = App.get_running_app()
        app.root.ids.status.text = "Exporting dataset..."

        def exportFunc():
            try:
                evaluate = Evaluate(self.settings)
                evaluate.export_dataset(fname)
            except Exception as exc:
                app.root.ids.status.text = str(exc)
            app.root.ids.status.text = "Finished exporting dataset"

        Thread(target=exportFunc).start()
        self.dismiss_popup()

    def show_export_fit_data_dialog(self):
        content = SaveDialog(save=self.export_fit_data,
                             cancel=self.dismiss_popup,
                             fname='fitData.csv')

        self._pop_up = Popup(title="Export Fit Data",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_export_settings_dialog(self):
        content = SaveDialog(save=self.export_settings,
                             cancel=self.dismiss_popup,
                             fname='cleaseSettings.json')

        self._pop_up = Popup(title="Export Settings",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def export_settings(self, path, selection, user_filename):
        if len(selection) == 0:
            fname = str(Path(path) / user_filename)
        else:
            fname = selection[0]

        if self.settings is None:
            return
        self.settings.save(fname)
        app = App.get_running_app()
        app.root.ids.status.text = f"Settings written to {fname}"
        self.dismiss_popup()


class CleaseGUI(App):

    def __init__(self):
        App.__init__(self)
        self.settings = None

    def build(self):
        self.icon = 'clease_logo.png'
        Window.bind(on_keyboard=self.on_keyboard)
        return WindowFrame()

    def on_keyboard(self, window, key, scancode, codepoint, modifier):
        if modifier in [['ctrl'], ['meta']] and codepoint == 's':
            self.root.save_session_to_current_file()

    def on_stop(self):
        for k, v in self.root.subprocesses.items():
            os.kill(v, signal.SIGTERM)
        self.root.subprocesses = {}


if __name__ == "__main__":
    CleaseGUI().run()
