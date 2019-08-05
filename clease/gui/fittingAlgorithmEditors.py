from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
import os


class FitAlgEditor(FloatLayout):
    close = ObjectProperty(None)
    backup_file = 'fit_alg_editor.txt'
    backup_folder = '.cleaseGUI/'

    def __init__(self, **kwargs):
        FloatLayout.__init__(self, **kwargs)
        if not os.path.exists(self.backup_folder):
            os.mkdir(self.backup_folder)
        self.load_values()

    @property
    def backup_filename(self):
        return self.backup_folder + self.backup_file

    def backup(self):
        values = []
        for child in self.ids.mainLayout.children:
            if isinstance(child, (TextInput, Spinner)):
                txt = child.text
                values.append(txt)

        with open(self.backup_filename, 'w') as out:
            for v in values:
                out.write('{}\n'.format(v))

    def load_values(self):
        try:
            with open(self.backup_filename, 'r') as infile:
                lines = infile.readlines()
            lines = [line.strip() for line in lines]

            counter = 0
            for child in self.ids.mainLayout.children:
                if isinstance(child, (TextInput, Spinner)):
                    child.text = lines[counter]
                    counter += 1
        except Exception:
            pass


class LassoEditor(FitAlgEditor):
    backup_file = 'lasso_editor.txt'


class L2Editor(FitAlgEditor):
    backup_file = 'l2_editor.txt'


class BCSEditor(FitAlgEditor):
    backup_file = 'bcs_editor.txt'


class GAEditor(FitAlgEditor):
    backup_file = 'ga_editor.txt'
