from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.spinner import Spinner
from kivy.uix.button import Button
from kivy.app import App
from kivy.clock import Clock
from threading import Thread
from clease.gui.util import parse_grouped_basis_elements, parse_elements
from clease.gui.util import parse_cell, parse_coordinate_basis, parse_cellpar
from clease.gui.util import parse_size


class SettingsInitialiser(object):
    """
    Class to be able to perform settings initialisation on a separate
    thread.
    """
    type = 'CEBulk'
    kwargs = None
    app = None
    status = None

    def initialise(self):
        from clease import CEBulk, CECrystal
        try:
            if self.type == 'CEBulk':
                self.app.settings = CEBulk(**self.kwargs)
            elif self.type == 'CECrystal':
                self.app.settings = CECrystal(**self.kwargs)
            self.status.text = 'Database initialised'
        except AssertionError as exc:
            self.status.text = "AssertError during initialization " + str(exc)
        except Exception as exc:
            self.status.text = str(exc)


class ConcentrationPage(Screen):
    next_constraint_id = 0
    elements = []
    grouped_elements = []
    grouped_basis = None

    def _elements_changed(self, new_elements):
        if len(self.grouped_elements) != len(new_elements):
            return True

        if any(len(a) != len(b) for a, b in zip(self.grouped_elements,
                                                new_elements)):
            return True

        return any(any(symb1 != symb2 for symb1, symb2 in zip(a, b))
                   for a, b in zip(self.grouped_elements, new_elements))

    def on_pre_enter(self):
        elem_str = self.manager.get_screen("Input").ids.elementInput.text
        grouped_basis = \
            self.manager.get_screen("Input").ids.groupedBasisInput.text
        try:
            if grouped_basis != '':
                self.grouped_basis = \
                    parse_grouped_basis_elements(grouped_basis)

            elements = parse_elements(elem_str)
        except Exception as exc:
            self.ids.status.text = str(exc)
            return
        new_elements = self._group_elements(elements, self.grouped_basis)

        if self._elements_changed(new_elements):
            for child in self.ids.mainConcLayout.children[:]:
                if child.id is None:
                    continue
                if child.id.startswith('cnst'):
                    self.ids.mainConcLayout.remove_widget(child)
                elif child.id == 'elemHeader':
                    self.ids.mainConcLayout.remove_widget(child)

            self.grouped_elements = new_elements
            self.elements = elements


            layout = StackLayout(id='elemHeader',size_hint=[1, 0.05])
            width = 1.0 / float(self.num_conc_vars + 3)

            for item in self.grouped_elements:
                for sym in item:
                    layout.add_widget(Label(text=sym, size_hint=[width, 1]))

            layout.add_widget(Label(text='Type',size_hint=[width, 1]))
            layout.add_widget(Label(text='Rhs',size_hint=[width, 1]))
            layout.add_widget(Label(text='Remove',size_hint=[width, 1]))
            self.ids.mainConcLayout.add_widget(layout)

    def _group_elements(self, elements, grouped_basis):
        if grouped_basis is None:
            return elements

        gr_elements = []
        for group in grouped_basis:
            gr_elements.append(elements[group[0]])
        return gr_elements

    @property
    def num_conc_vars(self):
        return sum(len(item) for item in self.grouped_elements)

    def add_constraint(self):
        layout = StackLayout(id="cnst{}".format(self.next_constraint_id),
                             size_hint=[1, 0.05])
        width = 1.0 / float(self.num_conc_vars + 3)
        for i in range(self.num_conc_vars):
            layout.add_widget(TextInput(text='0', multiline=False,
                                        size_hint=[width, 1],
                                        id='conc{}'.format(i)))
        layout.add_widget(Spinner(text='<=', values=['<=', '>=', '='],
                                  id='comparisonSpinner',
                                  size_hint=[width, 1]))
        layout.add_widget(TextInput(text='0', size_hint=[width, 1],                                                 multiline=False, id='rhs'))
        layout.add_widget(
            Button(text='Remove',
                   size_hint=[width, 1],
                   on_press=lambda _: self.remove_constraint(layout)))

        self.ids.mainConcLayout.add_widget(layout)
        self.next_constraint_id += 1
        print('yes, {}'.format(self.next_constraint_id))
        print(layout)
        return layout

    def remove_constraint(self, widget):
        self.ids.mainConcLayout.remove_widget(widget)

    def check_user_input(self):
        for widget in self.ids.mainConcLayout.children:
            if widget.id is None:
                continue

            if widget.id.startswith('cnst'):
                for child in widget.children:
                    if child.id is None:
                        continue
                    if child.id.startswith('conc') or child.id == 'rhs':
                        try:
                            _ = float(child.text)
                        except Exception:
                            msg = "All constraints need to be float"
                            self.ids.status.text = msg
                            return 1
        return 0

    def get_constraint_matrices(self):
        if self.check_user_input() != 0:
            raise ValueError("Could not parse constraints")

        # Loop across widgets
        A_lb = []
        A_eq = []
        rhs_lb = []
        rhs_eq = []
        for widget in self.ids.mainConcLayout.children:
            if widget.id is None:
                continue

            if widget.id.startswith('cnst'):
                matrix_row, rhs, equality = self._extract_row(widget)

                if equality:
                    A_eq.append(matrix_row)
                    rhs_eq.append(rhs)
                else:
                    A_lb.append(matrix_row)
                    rhs_lb.append(rhs)
        return A_lb, rhs_lb, A_eq, rhs_eq

    def _extract_row(self, widget):
        row = []
        ids = []
        rhs = 0.0
        spinner_value = ''
        for child in widget.children:
            if child.id == 'comparisonSpinner':
                spinner_value = child.text

        if spinner_value == '':
            raise ValueError("Could not extract the spinner value")

        equality = spinner_value == '='
        sign = 1.0
        if spinner_value == '<=':
            sign = -1.0
        for child in widget.children:
            if child.id is None:
                continue

            if child.id.startswith('conc'):
                row.append(sign*float(child.text))
                ids.append(child.id)
            elif child.id == 'rhs':
                rhs = sign*float(child.text)

        zipped = sorted(zip(ids, row))
        row = [z[1] for z in zipped]
        return row, rhs, equality

    def init_settings_class(self):
        try:
            from clease import Concentration
            A_lb, rhs_lb, A_eq, rhs_eq = self.get_constraint_matrices()

            input_page = self.manager.get_screen("Input")
            if input_page.check_user_input() != 0:
                self.ids.status.text = 'Error in input. Check the Input page.'
                return

            inputPage = input_page.to_dict()

            conc = Concentration(basis_elements=self.elements, A_lb=A_lb,
                                 b_lb=rhs_lb, A_eq=A_eq, b_eq=rhs_eq,
                                 grouped_basis=self.grouped_basis)

            supercell_factor = int(inputPage['supercell_factor'])
            skewness_factor = int(inputPage['skewness_factor'])
            size = None

            if inputPage['cell_size'] == '':
                size = None
            else:
                size = parse_size(inputPage['cell_size'])

            initialiser = SettingsInitialiser()
            initialiser.app = App.get_running_app()
            initialiser.status = self.ids.status

            if inputPage["type"] == 'CEBulk':
                if inputPage['aParameter'] == '':
                    a = None
                else:
                    a = float(inputPage['aParameter'])

                if inputPage['cParameter'] == '':
                    c = None
                else:
                    c = float(inputPage['cParameter'])

                if inputPage['uParameter'] == '':
                    u = None
                else:
                    u = float(inputPage['uParameter'])
                kwargs = dict(
                    crystalstructure=inputPage['crystalstructure'], a=a,
                    c=c, u=u,
                    db_name=inputPage['db_name'], concentration=conc,
                    max_cluster_dia=float(inputPage['max_cluster_dia']),
                    max_cluster_size=int(inputPage['cluster_size']),
                    basis_function=inputPage['basis_function'],
                    size=size, supercell_factor=supercell_factor,
                    skew_threshold=skewness_factor
                )
                self.ids.status.text = "Initialising database..."
                initialiser.type = 'CEBulk'
                initialiser.kwargs = kwargs
                Thread(target=initialiser.initialise).start()
            else:
                if inputPage['cellpar'] == '':
                    cellpar = None
                else:
                    cellpar = parse_cellpar(inputPage['cellpar'])

                if inputPage['basis'] == '':
                    basis = None
                else:
                    basis = parse_coordinate_basis(inputPage['basis'])

                if inputPage['cell'] == '':
                    cell = None
                else:
                    cell = parse_cell(inputPage['cell'])

                sp = int(inputPage['spacegroup'])
                self.ids.status.text = "Initialising database..."
                kwargs = dict(
                    basis=basis, cellpar=cellpar, cell=cell,
                    max_cluster_dia=float(inputPage['max_cluster_dia']),
                    max_cluster_size=int(inputPage['cluster_size']),
                    basis_function=inputPage['basis_function'],
                    size=size, supercell_factor=supercell_factor,
                    skew_threshold=skewness_factor,
                    concentration=conc, db_name=inputPage['db_name'],
                    spacegroup=sp
                )
                initialiser.type = 'CECrystal'
                initialiser.kwargs = kwargs
                Thread(target=initialiser.initialise).start()
        except Exception as exc:
            self.ids.status.text = str(exc)
            return

    def to_dict(self):
        A_lb, rhs_lb, A_eq, rhs_eq = self.get_constraint_matrices()
        data = {
            'A_lb': A_lb,
            'rhs_lb': rhs_lb,
            'A_eq': A_eq,
            'rhs_eq': rhs_eq
        }
        return data

    def load_from_matrices(self, A_lb, rhs_lb, A_eq, rhs_eq):
        self.on_pre_enter()

        for i in range(len(A_lb)):
            layout = self.add_constraint()

            # Traverse the children
            for child in layout.children:
                if child.id is None:
                    continue
                if child.id.startswith('conc'):
                    col = int(child.id[-1])
                    child.text = str(A_lb[i][col])
                elif child.id == 'rhs':
                    child.text = str(abs(rhs_lb[i]))
                elif child.id == 'comparisonSpinner':
                    if rhs_lb[i] < 0.0:
                        child.text = '<='
                    else:
                        child.text = '>='

        for i in range(len(A_eq)):
            layout = self.add_constraint()

            # Traverse children
            for child in layout.children:
                if child.id is None:
                    continue
                if child.id.startswith('conc'):
                    col = int(child.id[-1])
                    child.text = str(A_eq[i][col])
                elif child.id == 'rhs':
                    child.text = str(abs(rhs_eq[i]))
                elif child.id == 'comparisonSpinner':
                    child.text = '='

        # Initialise settings class
        self.init_settings_class()
