from kivy.uix.screenmanager import Screen
from kivy.uix.popup import Popup
from kivy.utils import get_color_from_hex

from clease.gui.constants import INACTIVE_TEXT_COLOR, FOREGROUND_TEXT_COLOR
from clease.gui.load_save_dialog import LoadDialog, SaveDialog
from clease.gui.util import parse_max_cluster_dia, parse_grouped_basis_elements
from clease.gui.util import parse_size, parse_elements, parse_cellpar
from clease.gui.util import parse_cell, parse_coordinate_basis
import json


class InputPage(Screen):
    cebulk_input_backup = {
        'crystStructSpinner': ''
    }
    current_session_file = None
    _pop_up = None

    def dismiss_popup(self):
        if self._pop_up is None:
            return
        self._pop_up.dismiss()
        self._pop_up = None

    def show_load_dialog(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._pop_up = Popup(title="Load structure DB", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

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

            self.ids.typeSpinner.text = data['type']
            self.ids.bfSpinner.text = data['basis_function']
            self.ids.clusterSize.text = data['cluster_size']
            self.ids.clusterDia.text = data['max_cluster_dia']
            self.db_name = data['db_name']
            self.ids.dbNameInput.text = self.db_name
            self.ids.crystStructSpinner.text = data['crystalstructure']
            self.ids.aParameterInput.text = data['aParameter']
            self.ids.cParameterInput.text = data['cParameter']
            self.ids.uParameterInput.text = data['uParameter']
            self.ids.cellParInput.text = data['cellpar']
            self.ids.cellInput.text = data['cell']
            self.ids.elementInput.text = data['elements']
            self.ids.crdBasisInput.text = data['basis']
            self.ids.spInput.text = data['spacegroup']
            self.ids.sizeInput.text = data.get('cell_size', '3, 3, 3')
            self.ids.sizeSpinner.text = data.get('cell_mode_spinner', 'Fixed')
            self.ids.scFactorInput.text = data.get('supercell_factor', '20')
            self.ids.skewFactorInput.text = data.get('skewness_factor', '4')
            self.ids.groupedBasisInput.text = data.get('grouped_basis', '')

            self.ids.status.text = "Loaded session from {}".format(path)

            A_lb = data['constraint']['A_lb']
            rhs_lb = data['constraint']['rhs_lb']
            A_eq = data['constraint']['A_eq']
            rhs_eq = data['constraint']['rhs_eq']
            self.manager.get_screen('Concentration').load_from_matrices(
                A_lb, rhs_lb, A_eq, rhs_eq)
            self.manager.get_screen('NewStruct').from_dict(
                data.get('new_struct', {}))
            self.manager.get_screen('Fit').from_dict(data.get('fit_page', {}))
            self.current_session_file = filename[0]
        except Exception as e:
            self.ids.status.text = "An error occured during load: " + str(e)
        self.dismiss_popup()

    def show_save_dialog(self):
        if self.check_user_input() != 0:
            return
        content = SaveDialog(save=self.save_session, cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Save CLEASE session", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def to_dict(self):
        return {
            'type': self.ids.typeSpinner.text,
            'basis_function': self.ids.bfSpinner.text,
            'cluster_size': self.ids.clusterSize.text,
            'max_cluster_dia': self.ids.clusterDia.text,
            'db_name': self.ids.dbNameInput.text,
            'crystalstructure': self.ids.crystStructSpinner.text,
            'aParameter': self.ids.aParameterInput.text,
            'cParameter': self.ids.cParameterInput.text,
            'uParameter': self.ids.uParameterInput.text,
            'cellpar': self.ids.cellParInput.text,
            'cell': self.ids.cellInput.text,
            'elements': self.ids.elementInput.text,
            'basis': self.ids.crdBasisInput.text,
            'spacegroup': self.ids.spInput.text,
            'cell_size': self.ids.sizeInput.text,
            'cell_mode_spinner': self.ids.sizeSpinner.text,
            'supercell_factor': self.ids.scFactorInput.text,
            'skewness_factor': self.ids.skewFactorInput.text,
            'grouped_basis': self.ids.groupedBasisInput.text
        }

    def save_session(self, path, selection, user_filename):
        if self.check_user_input() != 0:
            return
        if len(selection) == 0:
            fname = path + '/' + user_filename
        else:
            fname = selection[0]

        data = self.to_dict()
        data['constraint'] = self.manager.get_screen('Concentration').to_dict()
        data['new_struct'] = self.manager.get_screen('NewStruct').to_dict()
        data['fit_page'] = self.manager.get_screen('Fit').to_dict()

        with open(fname, 'w') as outfile:
            json.dump(data, outfile, separators=(',', ': '), indent=2)

        self.ids.status.text = 'Session saved to {}'.format(fname)
        self.dismiss_popup()
        self.current_session_file = fname

    def save_session_to_current_file(self):
        if self.current_session_file is None:
            self.ids.status.text = "No session file. Try Save As instead"
            return
        self.save_session(None, [self.current_session_file], None)

    def load(self, path, filename):
        self.db_path = path

        if len(filename) == 0:
            self.ids.dbNameInput.text = path
        else:
            self.ids.dbNameInput.text = filename[0]
        self.dismiss_popup()

    def update_input_section(self, text):
        if text == 'CECrystal':
            self._disable_CEBulk()
            self._enable_CECrystal()
        else:
            self._enable_CEBulk()
            self._disable_CECrystal()

    def on_enter(self):
        self.update_input_section(self.ids.typeSpinner.text)
        self.update_size_section(self.ids.sizeSpinner.text)

    def _disable_CEBulk(self):
        color = get_color_from_hex(INACTIVE_TEXT_COLOR)
        self.ids.CEBulkTitle.color = color
        self.ids.crystStruct.color = color
        self.ids.aParameter.color = color
        self.ids.cParameter.color = color
        self.ids.uParameter.color = color

        # Disable
        self.ids.crystStructSpinner.disabled = True
        self.ids.aParameterInput.disabled = True
        self.ids.cParameterInput.disabled = True
        self.ids.uParameterInput.disabled = True

    def _disable_CECrystal(self):
        color = get_color_from_hex(INACTIVE_TEXT_COLOR)
        self.ids.CECrystalTitle.color = color
        self.ids.cell.color = color
        self.ids.cellPar.color = color
        self.ids.spLabel.color = color
        self.ids.crBasisLabel.color = color

        # Disable fields
        self.ids.cellInput.disabled = True
        self.ids.cellParInput.disabled = True
        self.ids.spInput.disabled = True
        self.ids.crdBasisInput.disabled = True

    def _enable_CEBulk(self):
        color = get_color_from_hex(FOREGROUND_TEXT_COLOR)
        self.ids.CEBulkTitle.color = color
        self.ids.crystStruct.color = color
        self.ids.aParameter.color = color
        self.ids.cParameter.color = color
        self.ids.uParameter.color = color

        # Enable
        self.ids.crystStructSpinner.disabled = False
        self.ids.aParameterInput.disabled = False
        self.ids.cParameterInput.disabled = False
        self.ids.uParameterInput.disabled = False

    def _enable_CECrystal(self):
        color = get_color_from_hex(FOREGROUND_TEXT_COLOR)
        self.ids.CECrystalTitle.color = color
        self.ids.cell.color = color
        self.ids.cellPar.color = color
        self.ids.spLabel.color = color
        self.ids.crBasisLabel.color = color

        # Disable fields
        self.ids.cellInput.disabled = False
        self.ids.cellParInput.disabled = False
        self.ids.spInput.disabled = False
        self.ids.crdBasisInput.disabled = False

    def _check_cebulk_parameters(self):
        """
        Check the user input of the CE bulk parameters
        """
        try:
            _ = float(self.ids.aParameterInput.text)
        except Exception:
            self.ids.status.text = "a has to be a float"
            return 1

        c = self.ids.cParameterInput.text
        if c != '':
            try:
                _ = float(c)
            except Exception:
                self.ids.status.text = "c has to be float"
                return 1

        u = self.ids.uParameterInput.text

        if u != '':
            try:
                _ = float(u)
            except Exception:
                self.ids.status.text = "u has to be float"
                return 1
        return 0

    def cellpar_ok(self):
        cellPar = self.ids.cellParInput.text
        try:
            _ = parse_cellpar(cellPar)
        except Exception as exc:
            self.ids.status.text = str(exc)
            return False
        return True

    def cell_ok(self):
        cell = self.ids.cellInput.text
        try:
            _ = parse_cell(cell)
        except Exception as exc:
            self.ids.status.text = str(exc)
            return False
        return True

    def elem_ok(self):
        elems = self.ids.elementInput.text
        try:
            _ = parse_elements(elems)
        except Exception as exc:
            self.ids.status.text = str(exc)
            return False
        return True

    def grouped_basis_ok(self):
        gr_basis = self.ids.groupedBasisInput.text
        try:
            _ = parse_grouped_basis_elements(gr_basis)
        except Exception as exc:
            self.ids.status.text = str(exc)
            return False
        return True

    def _check_cecrystal_input(self):
        cellPar = self.ids.cellParInput.text
        sufficient_cell_info_given = False
        if cellPar != '':
            if not self.cellpar_ok():
                return 1
            sufficient_cell_info_given = True

        cell = self.ids.cellInput.text
        if cell != '':
            if not self.cell_ok():
                return 1
            sufficient_cell_info_given = True

        if not sufficient_cell_info_given:
            self.ids.status.text = 'Either cellpar or cell has to be given'
            return 1

        try:
            _ = parse_coordinate_basis(self.ids.crdBasisInput.text)
        except Exception as exc:
            self.ids.status.text = str(exc)
            return 1

        try:
            _ = int(self.ids.spInput.text)
        except Exception:
            self.ids.status.text = "Spacegroup has to be an integer"
            return 1
        return 0

    def max_cluster_dia_ok(self):
        cluster_size = int(self.ids.clusterSize.text)
        try:
            diameter = parse_max_cluster_dia(self.ids.clusterDia.text)

            if isinstance(diameter, list):
                if len(diameter) != cluster_size - 1:
                    self.ids.status.text = \
                        'Cluster dia has to be given for 2-body and beyond!'
                    return False
        except Exception as exc:
            self.ids.status.text = str(exc)
            return False
        return True

    def cell_size_ok(self):
        """Check if the cell size is OK."""
        try:
            _ = parse_size(self.ids.sizeInput.text)
        except Exception as exc:
            self.ids.status.text = str(exc)
            return False
        return True

    def check_user_input(self):
        """Check the input values from the user."""

        # Check max cluster size
        try:
            _ = int(self.ids.clusterSize.text)
        except Exception:
            self.ids.status.text = "Max cluster size has to be an integer"
            return 1

        # Check that we can parse the max cluster diameter
        if not self.max_cluster_dia_ok():
            return 1

        # Check that we can parse size
        if self.ids.sizeSpinner.text == 'Fixed':
            if not self.cell_size_ok():
                return 1
        else:
            if self.ids.scFactorInput.text == '':
                self.ids.status.text = 'Supercell factor has to be given'
                return 1

            if self.ids.skewFactorInput.text == '':
                self.ids.status.text = 'Skewness factor has to be given'
                return 1

        db_name = self.ids.dbNameInput.text
        if db_name == '':
            self.ids.status.text = "No database given"
            return 1

        if self.ids.typeSpinner.text == "CEBulk":
            error_code = self._check_cebulk_parameters()

            if error_code != 0:
                return error_code
        else:
            error_code = self._check_cecrystal_input()
            if error_code != 0:
                return error_code

        elems = self.ids.elementInput.text
        if elems == '':
            self.ids.status.text = 'No elements are given'
            return 1

        if not self.elem_ok():
            return 1

        gr_basis = self.ids.groupedBasisInput.text

        if gr_basis != '':
            if not self.grouped_basis_ok():
                return 1
        return 0

    def update_size_section(self, text):
        inactive = get_color_from_hex(INACTIVE_TEXT_COLOR)
        active = get_color_from_hex(FOREGROUND_TEXT_COLOR)
        if text == 'Fixed':
            self.ids.skewnessLabel.color = inactive
            self.ids.supercellFactLabel.color = inactive
            self.ids.sizeLabel.color = active

            self.ids.scFactorInput.disabled = True
            self.ids.skewFactorInput.disabled = True
            self.ids.sizeInput.disabled = False
        else:
            self.ids.skewnessLabel.color = active
            self.ids.supercellFactLabel.color = active
            self.ids.sizeLabel.color = inactive

            self.ids.scFactorInput.disabled = False
            self.ids.skewFactorInput.disabled = False
            self.ids.sizeInput.disabled = True
