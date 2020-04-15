from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
import os
import json
from ase.data import chemical_symbols
from clease import __version__ as version
from clease.gui import backup_folder


class SGCEditor(FloatLayout):
    close = ObjectProperty(None)
    data = {}
    backup_file = f"sgcEditorBck{version}.json"

    def __init__(self, **kwargs):
        symbols = kwargs.pop('symbols')
        chem_pot = kwargs.pop('chem_pot')
        FloatLayout.__init__(self, **kwargs)

        self.ids.symbolInput.text = symbols
        self.ids.chemPotInput.text = chem_pot

        if not backup_folder.exists():
            backup_folder.mkdir()
        self.load_values()

    def backup(self):
        self.data['symbols'] = self.ids.symbolInput.text
        self.data['chem_pot'] = self.ids.chemPotInput.text
        with open(self.fname, 'w') as f:
            json.dump(self.data, f)

    def load_values(self):
        if os.path.exists(self.fname):
            with open(self.fname, 'r') as f:
                self.data = json.load(f)

            self.ids.symbolInput.text = self.data['symbols']
            self.ids.chemPotInput.text = self.data['chem_pot']

    @property
    def fname(self):
        return str(backup_folder / self.backup_file)

    def symbols_are_valid(self, symbs):
        for s in symbs:
            if s not in chemical_symbols:
                return False
        return True

    def parse_symbols(self):
        symbs = self.ids.symbolInput.text
        remove = ['[', ']', '(', ')']
        for r in remove:
            symbs = symbs.replace(r, '')
        symbs = [x.strip() for x in symbs.split(',')]
        return symbs

    def parse_chem_pot(self):
        chem_pot = self.ids.chemPotInput.text
        chem_pot = chem_pot.replace('{', '')
        chem_pot = chem_pot.replace('}', '')

        chem_pot_dict = {}
        for item in chem_pot.split(','):
            splitted = item.split(':')
            if len(splitted) != 2:
                continue
            key = splitted[0].strip()
            value = float(splitted[1])
            chem_pot_dict[key] = value
        return chem_pot_dict
