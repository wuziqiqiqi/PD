from kivy.app import App
from kivy.lang import Builder
from kivy.uix.stacklayout import StackLayout
from kivy.resources import resource_add_path
from kivy.uix.popup import Popup

from clease.gui.settingsPage import SettingsPage
from clease.gui.concentrationPage import ConcentrationPage
from clease.gui.newStructPage import NewStructPage
from clease.gui.mc_page import MCPage
from clease.gui.fitPage import FitPage
from clease.gui.reconfigDB import ReconfigDB
from kivy.core.window import Window
from clease.gui.load_save_dialog import LoadDialog, SaveDialog

import json
from threading import Thread

import os.path as op

main_path = op.abspath(__file__)
main_path = main_path.rpartition("/")[0]
resource_add_path(main_path + '/layout')

Builder.load_file("cleaseGUILayout.kv")


class WindowFrame(StackLayout):
    _pop_up = None
    current_session_file = None
    settings = None
    reconfig_in_progress = False

    def dismiss_popup(self):
        if self._pop_up is None:
            return
        self._pop_up.dismiss()
        self._pop_up = None

    def show_load_session_dialog(self):
        content = LoadDialog(load=self.load_session, cancel=self.dismiss_popup)
        self._pop_up = Popup(title="Load CLEASE session", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
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

            fit_page = self.ids.sm.get_screen('Fit')
            fit_page.from_dict(data.get('fit_page', {}))

            mc_page = self.ids.sm.get_screen('MC')
            mc_page.from_dict(data.get('mc_page', {}))
            self.current_session_file = filename[0]

            msg = "Loaded session from {}".format(self.current_session_file)
            App.get_running_app().root.ids.status.text = msg

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
            fname = path + '/' + user_filename
        else:
            fname = selection[0]

        data = {}
        data['settings'] = self.ids.sm.get_screen('Settings').to_dict()
        data['conc'] = self.ids.sm.get_screen('Concentration').to_dict()
        data['new_struct'] = self.ids.sm.get_screen('NewStruct').to_dict()
        data['fit_page'] = self.ids.sm.get_screen('Fit').to_dict()
        data['mc_page'] = self.ids.sm.get_screen('MC').to_dict()

        with open(fname, 'w') as outfile:
            json.dump(data, outfile, separators=(',', ': '), indent=2)
        msg = 'Session saved to {}'.format(fname)
        App.get_running_app().root.ids.status.text = msg
        self.dismiss_popup()
        self.current_session_file = fname

    def show_save_dialog(self):
        if self.ids.sm.get_screen('Settings').check_user_input() != 0:
            return

        if self.ids.sm.get_screen('Concentration').check_user_input() != 0:
            return

        content = SaveDialog(save=self.save_session, cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Save CLEASE session", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
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
        self.ids.sm.current = new_screen

    def reconfig(self, target=None):
        """Reconfigure target.

        Parameters:

        target: str
            one of "settings", "db" and "settings_db"
        """
        if self.reconfig_in_progress:
            # Do no allow user to initialize many threads
            return

        self.reconfig_in_progress = True

        reconfig = ReconfigDB()
        reconfig.app = App.get_running_app()
        reconfig.status = App.get_running_app().root.ids.status

        if target == 'settings':
            Thread(target=reconfig.reconfig_settings).start()
        elif target == 'db':
            Thread(target=reconfig.reconfig_db).start()
        else:
            Thread(target=reconfig.reconfig_settings_db).start()

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
            self.ids.sm.get_screen('Settings').apply_settings()

        try:
            self.settings._activate_lagest_template()
            atoms = self.settings.atoms
            return self.settings.cluster_list.get_figures(atoms)
        except Exception:
            return False


class CleaseGUI(App):
    def __init__(self):
        App.__init__(self)
        self.settings = None

    def build(self):
        self.icon = 'clease_logo.png'
        Window.bind(on_keyboard=self.on_keyboard)
        return WindowFrame()

    def on_keyboard(self, window, key, scancode, codepoint, modifier):
        if modifier == ['ctrl'] and codepoint == 's':
            self.root.save_session_to_current_file()


if __name__ == "__main__":
    CleaseGUI().run()
