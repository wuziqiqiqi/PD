from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from clease import __version__ as version
import os
import json
from clease.gui import backup_folder


class ConcObsEditor(FloatLayout):
    close = ObjectProperty(None)
    backup_file = f"concObsBackup{version}.json"
    data = {}
    default_element = 'X'

    def __init__(self, **kwargs):
        self.default_element = kwargs.pop('default_element')
        FloatLayout.__init__(self, **kwargs)
        if not backup_folder.exists():
            backup_folder.mkdir()
        self.load_values()
        self._init_fields()

    @property
    def fname(self):
        return str(backup_folder / self.backup_file)

    def load_values(self):
        if os.path.exists(self.fname):
            with open(self.fname, 'r') as f:
                self.data = json.load(f)

    def _init_fields(self):
        element = self.data.get('element', self.default_element)
        if element == '':
            element = self.default_element

        self.ids.elementInput.text = element

    def backup(self):
        self.data['element'] = self.ids.elementInput.text
        with open(self.fname, 'w') as f:
            json.dump(self.data, f)
