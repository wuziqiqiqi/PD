from kivy.uix.screenmanager import Screen
from kivy.uix.popup import Popup
from kivy.properties import StringProperty
from clease.gui.help_message_popup import HelpMessagePopup
from kivy.app import App
from clease.gui.load_save_dialog import LoadDialog


class MCMainPage(Screen):
    _pop_up = None
    mc_cell_lengths = StringProperty('a: b: c:')
    angle_info = StringProperty('alpha: beta: gamma:')
    num_atoms = StringProperty('No. atoms')
    eci_file = StringProperty('')
    mc_cell_size = StringProperty('1')

    def dismiss_popup(self):
        self._pop_up.dismiss()
        self._pop_up = None

    @property
    def main_mc_screen(self):
        app = App.get_running_app()
        return app.root.ids.sm.get_screen('MCHeader')

    @property
    def info_update_disabled(self):
        app = App.get_running_app()
        return app.root.info_update_disabled

    def show_size_help_message(self):
        msg = 'Scaling factor used to scale the currently active template\n'
        msg += 'If this factor is 5 a supercell consiting (5, 5, 5)\n'
        msg += 'extension of the active template will be used for the\n'
        msg += 'MC calculation\n'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Concentration input",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def get_mc_cell(self):
        app = App.get_running_app()
        return app.root.get_mc_cell()

    def set_cell_info(self):
        if self.info_update_disabled:
            return

        atoms = self.get_mc_cell()
        if atoms is None:
            return

        info = atoms.get_cell_lengths_and_angles()

        length_str = f"a: {int(info[0])} Å b: {int(info[1])} Å "
        length_str += f"c: {int(info[1])} Å"
        self.mc_cell_lengths = length_str

        angle_str = f"{int(info[3])} deg {int(info[4])} deg {int(info[5])} deg"
        self.angle_info = angle_str
        self.num_atoms = f"No. atoms {len(atoms)}"

    def show_load_dialog(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._pop_up = Popup(title="Load ECIs",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def load(self, path, filename):
        if len(filename) == 0:
            self.eci_file = path
        else:
            self.eci_file = filename[0]
        self.dismiss_popup()

    def to_dict(self):
        return {'size': self.ids.sizeInput.text, 'eci_file': self.ids.eciFileInput.text}

    def from_dict(self, dct):
        self.ids.sizeInput.text = dct.get('size', '1')
        self.ids.eciFileInput.text = dct.get('eci_file', '')
