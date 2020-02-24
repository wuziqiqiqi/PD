from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty, StringProperty
from clease.gui.util import parse_select_cond
from ase.db import connect
from copy import deepcopy


class DbBrowser(FloatLayout):
    close = ObjectProperty(None)
    text = StringProperty('')

    def __init__(self, **kwargs):
        self.db = connect(kwargs.pop('db_name'))
        self.spacing = self._format_constraints()
        FloatLayout.__init__(self, **kwargs)
        txt = f"| {'id':{self.spacing['id']}} |"
        txt += f" {'formula':{self.spacing['formula']}} |"
        txt += f" {'calculator':{self.spacing['calc']}} |"
        txt += f" {'energy':{self.spacing['energy']}} |"
        txt += f" {'name':{self.spacing['name']}} |"
        txt += f" {'size':{self.spacing['size']}} |"
        txt += f" {'gen':{self.spacing['gen']}} |"
        txt += f" {'struct_type':{self.spacing['struct_type']}} |"
        txt += f" {'converged':9} |"
        self.header = txt
        self.set_rows(None)

    def _format_row(self, row):
        calc = row.get('calculator', '')
        name = row.get('name', 'NN')
        gen = row.get('gen', '')
        energy = row.get('energy', '')
        str_type = row.get('struct_type', '')
        cnv = row.get('converged', '')
        size = row.get('size', '')
        if isinstance(energy, float):
            energy = f"{energy:.3f}"

        if isinstance(cnv, bool):
            cnv = int(cnv)
        txt = f"| {str(row.id):{self.spacing['id']}} |"
        txt += f" {row.formula:{self.spacing['formula']}} |"
        txt += f" {calc:{self.spacing['calc']}} |"
        txt += f" {energy:{self.spacing['energy']}} |"
        txt += f" {name:{self.spacing['name']}} |"
        txt += f" {size:{self.spacing['size']}} |"
        txt += f" {str(gen):{self.spacing['gen']}} |"
        txt += f" {str_type:{self.spacing['struct_type']}} |"
        txt += f" {str(cnv):9} |"
        return txt + '\n'

    def _format_constraints(self):
        spacing = {
            'id': 2,
            'formula': 7,
            'calc': 10,
            'energy': 6,
            'name': 4,
            'gen': 3,
            'struct_type': 11,
            'size': 4
        }

        for row in self.db.select():
            for k, v in spacing.items():
                value = row.get(k, 'none')

                if isinstance(value, float):
                    value = f"{value:.3f}"
                if len(str(value)) > v:
                    spacing[k] = len(str(value))
        return spacing

    def on_select(self, txt):
        select_cond = parse_select_cond(txt)
        self.set_rows(select_cond)

    def set_rows(self, select_cond):
        all_str = deepcopy(self.header) + '\n'

        if select_cond is None:
            db_iterator = self.db.select()
        else:
            db_iterator = self.db.select(select_cond)
        for row in db_iterator:
            str_rep = self._format_row(row)
            all_str += str_rep

        self.text = all_str
