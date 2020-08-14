from kivy.uix.screenmanager import Screen
from kivy.uix.popup import Popup
from kivy.utils import get_color_from_hex
from kivy.app import App

from clease.gui.constants import INACTIVE_TEXT_COLOR, FOREGROUND_TEXT_COLOR
from clease.gui.load_save_dialog import LoadDialog
from clease.gui.util import parse_max_cluster_dia, parse_size
from clease.gui.util import parse_cell, parse_coordinate_basis, parse_cellpar
from clease.gui.settings_initializer import SettingsInitializer
from threading import Thread
import traceback


class SettingsPage(Screen):
    _pop_up = None

    def dismiss_popup(self):
        if self._pop_up is None:
            return
        self._pop_up.dismiss()
        self._pop_up = None

    def show_load_dialog(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._pop_up = Popup(title="Load structure DB",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
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
            'basis': self.ids.crdBasisInput.text,
            'spacegroup': self.ids.spInput.text,
            'cell_size': self.ids.sizeInput.text,
            'cell_mode_spinner': self.ids.sizeSpinner.text,
            'supercell_factor': self.ids.scFactorInput.text,
            'skew_threshold': self.ids.skewThresholdInput.text
        }

    def from_dict(self, data):
        self.ids.typeSpinner.text = data['type']
        self.ids.bfSpinner.text = data['basis_function']
        self.ids.clusterSize.text = data['cluster_size']
        self.ids.clusterDia.text = data['max_cluster_dia']
        self.db_name = data['db_name']
        self.ids.dbNameInput.text = data['db_name']
        self.ids.crystStructSpinner.text = data['crystalstructure']
        self.ids.aParameterInput.text = data['aParameter']
        self.ids.cParameterInput.text = data['cParameter']
        self.ids.uParameterInput.text = data['uParameter']
        self.ids.cellParInput.text = data['cellpar']
        self.ids.cellInput.text = data['cell']
        self.ids.crdBasisInput.text = data['basis']
        self.ids.spInput.text = data['spacegroup']
        self.ids.sizeInput.text = data.get('cell_size', '3, 3, 3')
        self.ids.sizeSpinner.text = data.get('cell_mode_spinner', 'Fixed')
        self.ids.scFactorInput.text = data.get('supercell_factor', '20')
        self.ids.skewThresholdInput.text = data.get('skew_threshold', '40')

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
        """Check the user input of the CE bulk parameters."""
        try:
            _ = float(self.ids.aParameterInput.text)
        except Exception:
            App.get_running_app().root.ids.status.text =\
                "a has to be a float"
            return 1

        c = self.ids.cParameterInput.text
        if c != '':
            try:
                _ = float(c)
            except Exception:
                App.get_running_app().root.ids.status.text =\
                    "c has to be float"
                return 1

        u = self.ids.uParameterInput.text

        if u != '':
            try:
                _ = float(u)
            except Exception:
                App.get_running_app().root.ids.status.text =\
                    "u has to be float"
                return 1
        return 0

    def cellpar_ok(self):
        cellPar = self.ids.cellParInput.text
        try:
            _ = parse_cellpar(cellPar)
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
            return False
        return True

    def cell_ok(self):
        cell = self.ids.cellInput.text
        try:
            _ = parse_cell(cell)
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
            return False
        return True

    def coordinate_basis_ok(self):
        crdBasis = self.ids.crdBasisInput.text
        try:
            _ = parse_coordinate_basis(crdBasis)
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
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
            msg = 'Either cellpar or cell has to be given'
            App.get_running_app().root.ids.status.text = msg
            return 1

        if not self.coordinate_basis_ok():
            return 1

        try:
            _ = int(self.ids.spInput.text)
        except Exception:
            msg = "Spacegroup has to be an integer."
            App.get_running_app().root.ids.status.text = msg
            return 1
        return 0

    def max_cluster_dia_ok(self):
        cluster_size = int(self.ids.clusterSize.text)
        try:
            diameter = parse_max_cluster_dia(self.ids.clusterDia.text)
            if isinstance(diameter, list):
                if len(diameter) != cluster_size - 1:
                    msg = 'Cluster dia has to be given for 2-body and beyond!'
                    App.get_running_app().root.ids.status.text = msg
                    return False
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
            return False
        return True

    def cell_size_ok(self):
        """Check if the cell size is OK."""
        try:
            _ = parse_size(self.ids.sizeInput.text)
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
            return False
        return True

    def check_user_input(self):
        """Check the input values from the user."""

        # Check max cluster size
        try:
            _ = int(self.ids.clusterSize.text)
        except Exception:
            msg = "Max cluster size has to be an integer."
            App.get_running_app().root.ids.status.text = msg
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
                msg = 'Supercell factor has to be given'
                App.get_running_app().root.ids.status.text = msg
                return 1

            if self.ids.skewThresholdInput.text == '':
                msg = 'Skewness factor has to be given'
                App.get_running_app().root.ids.status.text = msg
                return 1

        db_name = self.ids.dbNameInput.text
        if db_name == '':
            msg = "No database given"
            App.get_running_app().root.ids.status.text = msg
            return 1

        if self.ids.typeSpinner.text == "CEBulk":
            error_code = self._check_cebulk_parameters()

            if error_code != 0:
                return error_code
        else:
            error_code = self._check_cecrystal_input()
            if error_code != 0:
                return error_code

        return 0

    def update_size_section(self, text):
        inactive = get_color_from_hex(INACTIVE_TEXT_COLOR)
        active = get_color_from_hex(FOREGROUND_TEXT_COLOR)
        if text == 'Fixed':
            self.ids.skewnessLabel.color = inactive
            self.ids.supercellFactLabel.color = inactive
            self.ids.sizeLabel.color = active

            self.ids.scFactorInput.disabled = True
            self.ids.skewThresholdInput.disabled = True
            self.ids.sizeInput.disabled = False
        else:
            self.ids.skewnessLabel.color = active
            self.ids.supercellFactLabel.color = active
            self.ids.sizeLabel.color = inactive

            self.ids.scFactorInput.disabled = False
            self.ids.skewThresholdInput.disabled = False
            self.ids.sizeInput.disabled = True

    def apply_settings(self):
        try:
            from clease import Concentration
            conc_page = self.manager.get_screen("Concentration")
            if conc_page.check_user_input() != 0:
                return

            A_lb, rhs_lb, A_eq, rhs_eq = conc_page.get_constraint_matrices()

            if not conc_page.elements:
                msg = 'It appears that the Apply button in Concentration panel'
                msg += ' was not clicked.'
                App.get_running_app().root.ids.status.text = msg
                return
            basis_elements = conc_page.elements
            grouped_basis = conc_page.grouped_basis
            conc = Concentration(basis_elements=basis_elements,
                                 A_lb=A_lb,
                                 b_lb=rhs_lb,
                                 A_eq=A_eq,
                                 b_eq=rhs_eq,
                                 grouped_basis=grouped_basis)

            if self.check_user_input() != 0:
                return

            settingsPage = self.to_dict()
            supercell_factor = int(settingsPage['supercell_factor'])
            skew_threshold = int(settingsPage['skew_threshold'])
            size = None

            if self.ids.sizeSpinner.text == 'Fixed':
                size = parse_size(settingsPage['cell_size'])

            if settingsPage['max_cluster_dia'] == '':
                max_cluster_dia = None
            else:
                max_cluster_dia =\
                    parse_max_cluster_dia(settingsPage['max_cluster_dia'])

            initializer = SettingsInitializer()
            initializer.basis_func_type = settingsPage['basis_function']
            initializer.skew_threshold = skew_threshold
            initializer.app = App.get_running_app()
            initializer.status = App.get_running_app().root.ids.status

            if settingsPage["type"] == 'CEBulk':
                if settingsPage['aParameter'] == '':
                    a = None
                else:
                    a = float(settingsPage['aParameter'])

                if settingsPage['cParameter'] == '':
                    c = None
                else:
                    c = float(settingsPage['cParameter'])

                if settingsPage['uParameter'] == '':
                    u = None
                else:
                    u = float(settingsPage['uParameter'])
                kwargs = dict(
                    crystalstructure=settingsPage['crystalstructure'],
                    a=a,
                    c=c,
                    u=u,
                    db_name=settingsPage['db_name'],
                    concentration=conc,
                    max_cluster_dia=max_cluster_dia,
                    max_cluster_size=int(settingsPage['cluster_size']),
                    size=size,
                    supercell_factor=supercell_factor,
                )
                msg = "Applying settingss to database..."
                App.get_running_app().root.ids.status.text = msg
                initializer.type = 'CEBulk'
                initializer.kwargs = kwargs
                initializer.basis_function = settingsPage['basis_function']
                initializer.skew_threshold = skew_threshold
                Thread(target=initializer.initialize).start()
            else:
                if settingsPage['cellpar'] == '':
                    cellpar = None
                else:
                    cellpar = parse_cellpar(settingsPage['cellpar'])

                if settingsPage['basis'] == '':
                    basis = None
                else:
                    basis = parse_coordinate_basis(settingsPage['basis'])

                if settingsPage['cell'] == '':
                    cell = None
                else:
                    cell = parse_cell(settingsPage['cell'])

                sp = int(settingsPage['spacegroup'])
                msg = "Applying settingss to database..."
                App.get_running_app().root.ids.status.text = msg
                kwargs = dict(basis=basis,
                              cellpar=cellpar,
                              cell=cell,
                              max_cluster_dia=max_cluster_dia,
                              max_cluster_size=int(settingsPage['cluster_size']),
                              basis_function=settingsPage['basis_function'],
                              size=size,
                              supercell_factor=supercell_factor,
                              skew_threshold=skew_threshold,
                              concentration=conc,
                              db_name=settingsPage['db_name'],
                              spacegroup=sp)
                initializer.type = 'CECrystal'
                initializer.kwargs = kwargs
                Thread(target=initializer.initialize).start()
                msg = "Settings initialized."
                App.get_running_app().root.ids.status.text = msg
            return True

        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
            return False
