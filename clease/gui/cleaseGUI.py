from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout


from clease.gui.settingsPage import SettingsPage
from clease.gui.concentrationPage import ConcentrationPage
from clease.gui.newStructPage import NewStructPage
from clease.gui.fitPage import FitPage
from kivy.resources import resource_add_path

from clease.gui.load_save_dialog import LoadDialog, SaveDialog
from kivy.uix.popup import Popup
import json

import os.path as op

main_path = op.abspath(__file__)
main_path = main_path.rpartition("/")[0]
resource_add_path(main_path + '/layout')

Builder.load_file("cleaseGUILayout.kv")

class WindowFrame(BoxLayout):
    _pop_up = None
    current_session_file = None

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
            settings_page.apply_update_settings()

            newstruct_page = self.ids.sm.get_screen('NewStruct')
            newstruct_page.from_dict(data.get('new_struct', {}))

            fit_page = self.ids.sm.get_screen('Fit')
            fit_page.from_dict(data.get('fit_page', {}))
            self.current_session_file = filename[0]

            settings_page.ids.status.text = \
                "Loaded session from {}".format(self.current_session_file)

        except Exception as e:
            settings_page.ids.status.text = "An error occured during load: " + str(e)
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

        with open(fname, 'w') as outfile:
            json.dump(data, outfile, separators=(',', ': '), indent=2)
        self.ids.sm.get_screen('Settings').ids.status.text = \
            'Session saved to {}'.format(fname)
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


class CleaseGUI(App):
    def __init__(self):
        App.__init__(self)
        self.settings = None

    def build(self):
        self.icon = 'clease_logo.png'
        return WindowFrame()


if __name__ == "__main__":
    CleaseGUI().run()
