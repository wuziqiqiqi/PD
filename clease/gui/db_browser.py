import logging
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty, StringProperty
from ase.db import connect
from ase.formula import Formula
from copy import deepcopy
from typing import Tuple, List, Dict
import re
from ase.data import chemical_symbols

logger = logging.getLogger(__name__)

SelectCond = Tuple[List[dict], Dict[str, dict]]

# Kivy can show a black screen if there is too much information in a label
# Limit the view to 100 rows
MAX_ROWS = 100


class DbBrowser(FloatLayout):
    close = ObjectProperty(None)
    text = StringProperty('')

    def __init__(self, **kwargs) -> None:
        self.db_name = kwargs.pop('db_name')
        self.spacing = {
            'id': 2,
            'formula': 7,
            'calc': 10,
            'energy': 6,
            'name': 4,
            'gen': 3,
            'struct_type': 11,
            'size': 4
        }
        self._format_constraints()

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
        self.set_rows([], {})

    def _format_row(self, row: dict) -> str:
        rowId = row.get('id', -1)
        formula = row.get('formula', 'unknown')
        calc = row.get('calc', 'none') or 'none'
        name = row.get('name', 'NN') or 'NN'
        gen = int(row.get('gen', 0))
        energy = row.get('energy', '') or 0.0
        str_type = row.get('struct_type', '')
        cnv = int(row.get('converged', 0))
        size = row.get('size', '')
        if isinstance(energy, float):
            energy = f"{energy:.3f}"

        if isinstance(cnv, bool):
            cnv = int(cnv)
        txt = f"| {str(rowId):{self.spacing['id']}} |"
        txt += f" {formula:{self.spacing['formula']}} |"
        txt += f" {calc:{self.spacing['calc']}} |"
        txt += f" {energy:{self.spacing['energy']}} |"
        txt += f" {name:{self.spacing['name']}} |"
        txt += f" {size:{self.spacing['size']}} |"
        txt += f" {str(gen):{self.spacing['gen']}} |"
        txt += f" {str_type:{self.spacing['struct_type']}} |"
        txt += f" {str(cnv):9} |"
        return txt + '\n'

    def _update_spacing(self, row: dict) -> None:
        for k, v in row.items():
            default_length = self.spacing.get(k, 0)

            if isinstance(v, float):
                v = f"{v:.3f}"
            if len(str(v)) > default_length:
                self.spacing[k] = len(str(v))

    def _get_formulas(self, cur) -> Dict[int, str]:
        sql = 'SELECT Z, n, id FROM species'
        cur.execute(sql)
        count_dict = {}
        for item in cur.fetchall():
            rowId = item[2]
            current_count = count_dict.get(rowId, {})
            symb = chemical_symbols[item[0]]
            n = item[1]
            current_count[symb] = n
            count_dict[rowId] = current_count

        formula_dict = {}
        for rowId, count in count_dict.items():
            formula = Formula(_count=count)
            formula_dict[rowId] = formula.format('hill')
        return formula_dict

    def _format_constraints(self) -> None:
        with connect(self.db_name) as db:
            cur = db.connection.cursor()
            formulas = self._get_formulas(cur)
            sql = 'SELECT id, calculator, energy FROM systems'
            cur.execute(sql)
            for item in cur.fetchall():
                row = {
                    'id': item[0],
                    'calc': item[1],
                    'energy': item[2],
                    'formula': formulas[item[0]]
                }
                self._update_spacing(row)

            condition = 'key="name" OR key="size" OR key="struct_type"'
            sql = f'SELECT key, value FROM text_key_values WHERE {condition}'
            cur.execute(sql)
            for item in cur.fetchall():
                self._update_spacing({item[0]: item[1]})

            sql = f'SELECT key, value FROM number_key_values WHERE key="gen"'
            cur.execute(sql)
            for item in cur.fetchall():
                self._update_spacing({item[0]: item[1]})

    def _get_select_conditions(self, txt: str) -> SelectCond:
        system_constraints = []
        kvp_constraints = {}
        prog = re.compile(r"(\w+)[=<>!]*(\w+)")
        known_op = ['=', '>', '<', '>=', '<=', '!=']

        system_cols = [
            'id', 'calculator', 'energy', 'fmax', 'smax', 'natoms', 'user', 'volume', 'mass',
            'charge'
        ]

        # Construct selection conditions
        for cond in txt.split(','):
            cond = cond.strip()
            match = prog.match(cond)
            if match is None:
                continue
            key = match.group(1)
            if key == 'formula':
                # Formula is treated separately
                continue
            value = match.group(2)
            op = cond.replace(key, '')
            op = op.replace(value, '')

            # Check that the reminding part is actually a known operator
            # Should protect against undesired injections
            if op not in known_op:
                logger.warning(f"Unkown operator {cond}")
                continue

            if key in system_cols:
                system_constraints.append({'sql': key + op + '?', 'value': value})

            else:
                default = {'sql': [], 'values': []}
                current_constraint = kvp_constraints.get(key, default)
                current_constraint['sql'].append(f'value{op}?')
                current_constraint['values'].append(value)
                kvp_constraints[key] = current_constraint
        return system_constraints, kvp_constraints

    def on_select(self, txt: str) -> None:
        try:
            system_conditions, kvp_conditions = self._get_select_conditions(txt)
            formula = 'all'
            if 'formula' in txt:
                prog = re.compile(r"formula=(\w+)")
                m = prog.match(txt)
                if m is None:
                    logger.warning("Unsupported operation for formula")
                else:
                    formula = m.group(1)
            self.set_rows(system_conditions, kvp_conditions, formula=formula)
        except Exception as exc:
            logger.exception(exc)

    def _count_kvps(self, cur) -> Dict[int, int]:
        """
        Counts the number of key value pairs for each ID
        """
        tables = ['number_key_values', 'text_key_values']
        counter = {}
        for tab in tables:
            sql = f'select id from {tab}'
            cur.execute(sql)

            for rowId in cur.fetchall():
                counter[rowId] = counter.get(rowId, 0) + 1
        return counter

    def set_rows(self, system_conditions, kvp_conditions, formula='all'):
        all_str = deepcopy(self.header) + '\n'

        rows = {}
        with connect(self.db_name) as db:
            cur = db.connection.cursor()
            formulas = self._get_formulas(cur)

            # Go through the DB and find which IDs are valid according
            # to the constraints
            validIds = set()
            sql = 'SELECT id FROM systems'
            if system_conditions:
                sqls = [x['sql'] for x in system_conditions]
                values = [x['value'] for x in system_conditions]
                sql += ' WHERE ' + ' AND '.join(sqls)
                cur.execute(sql, values)
            else:
                cur.execute(sql)

            for rowId in cur.fetchall():
                validIds.add(rowId[0])

            # Limit valid ids based on kvp_conditions
            for k, v in kvp_conditions.items():
                satisfyConstraint = set()
                tables = ['number_key_values', 'text_key_values']
                for tab in tables:
                    sql = f'select id from {tab} WHERE '
                    sql += 'key=? AND ' + ' AND '.join(v['sql'])
                    cur.execute(sql, [k] + v['values'])
                    for rowId in cur.fetchall():
                        satisfyConstraint.add(rowId[0])
                validIds = validIds.intersection(satisfyConstraint)

            # Apply formula constraint
            if formula != 'all':
                formulaIds = set(k for k, v in formulas.items() if v == formula)
                validIds = validIds.intersection(formulaIds)

            # Extract information from systems that is consistent
            # with validIds
            sql = 'SELECT id, calculator, energy FROM systems'
            cur.execute(sql)
            for item in cur.fetchall():
                rowId = item[0]
                if rowId in validIds:
                    row = {
                        'id': item[0],
                        'formula': formulas[rowId],
                        'calc': item[1],
                        'energy': item[2]
                    }
                    rows[rowId] = row

            # Extract information from number_key_values that is consistent
            # with valid ids
            sql = 'select key, value, id from number_key_values WHERE '
            sql += 'key="gen" OR key="converged"'
            cur.execute(sql)
            for key, value, rowId in cur.fetchall():
                if rowId in validIds:
                    rows[rowId][key] = value

            # Extract information from text_key_values that is consistent
            # with valid ids
            sql = 'select key, value, id from text_key_values WHERE '
            sql += 'key="name" OR key="size" OR key="struct_type"'
            cur.execute(sql)
            for key, value, rowId in cur.fetchall():
                if rowId in validIds:
                    rows[rowId][key] = value

        row_count = 0
        for row in rows.values():
            str_rep = self._format_row(row)
            all_str += str_rep
            row_count += 1
            if row_count >= MAX_ROWS:
                break

        self.text = all_str
