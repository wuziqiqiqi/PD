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
        txt = '| {:{space}s} |'.format('id', space=self.spacing['id'])
        txt += ' {:{space}s} |'.format('formula', space=self.spacing['formula'])
        txt += ' {:{space}s} |'.format('calculator', space=self.spacing['calc'])
        txt += ' {:{space}s} |'.format('energy', space=self.spacing['energy'])
        txt += ' {:{space}s} |'.format('name', space=self.spacing['name'])
        txt += ' {:{space}s} |'.format('size', space=self.spacing['size'])
        txt += ' {:{space}s} |'.format('gen', space=self.spacing['gen'])
        txt += ' {:{space}s} |'.format('struct_type',
                                       space=self.spacing['struct_type'])
        txt += ' {:9s} |'.format('converged')
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
            energy = '{:.3f}'.format(energy)

        if isinstance(cnv, bool):
            cnv = int(cnv)
        txt = '| {:{space}s} |'.format(str(row.id), space=self.spacing['id'])
        txt += ' {:{space}s} |'.format(row.formula,
                                       space=self.spacing['formula'])
        txt += ' {:{space}s} |'.format(calc, space=self.spacing['calc'])
        txt += ' {:{space}s} |'.format(energy, space=self.spacing['energy'])
        txt += ' {:{space}s} |'.format(name, space=self.spacing['name'])
        txt += ' {:{space}s} |'.format(size, space=self.spacing['size'])
        txt += ' {:{space}s} |'.format(str(gen), space=self.spacing['gen'])
        txt += ' {:{space}s} |'.format(str_type,
                                       space=self.spacing['struct_type'])
        txt += ' {:9s} |'.format(str(cnv))
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
                    value = '{:.3f}'.format(value)
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
