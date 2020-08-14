from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.stacklayout import StackLayout
from kivy.uix.spinner import Spinner
from kivy.uix.button import Button
from kivy.app import App

from clease.gui.util import parse_grouped_basis_elements, parse_elements
import traceback


class ConcentrationPage(Screen):
    next_constraint_id = 0
    elements = []
    grouped_elements = []
    grouped_basis = None

    def _elements_changed(self, new_elements):
        if len(self.grouped_elements) != len(new_elements):
            return True

        if any(len(a) != len(b) for a, b in zip(self.grouped_elements, new_elements)):
            return True

        return any(
            any(symb1 != symb2
                for symb1, symb2 in zip(a, b))
            for a, b in zip(self.grouped_elements, new_elements))

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
        layout = StackLayout(id=f"cnst{self.next_constraint_id}", size_hint=[1, 0.05])
        width = 1.0 / float((self.num_conc_vars + 3))
        for i in range(self.num_conc_vars):
            layout.add_widget(
                TextInput(text='0',
                          size_hint=[width, 1],
                          write_tab=False,
                          multiline=False,
                          id=f"conc{i}"))
        layout.add_widget(
            Spinner(text='<=',
                    values=['<=', '>=', '='],
                    id='comparisonSpinner',
                    size_hint=[width, 1]))
        layout.add_widget(
            TextInput(text='0', size_hint=[width, 1], write_tab=False, multiline=False, id='rhs'))
        layout.add_widget(
            Button(text='Remove',
                   size_hint=[width, 1],
                   on_press=lambda _: self.remove_constraint(layout)))

        self.ids.mainConcLayout.add_widget(layout)
        self.next_constraint_id += 1
        return layout

    def remove_constraint(self, widget):
        self.ids.mainConcLayout.remove_widget(widget)

    def check_user_input(self):
        elems = self.ids.elementInput.text
        if elems == '':
            msg = 'No elements are given.'
            App.get_running_app().root.ids.status.text = msg
            return 1

        if not self.elem_ok():
            return 1

        gr_basis = self.ids.groupedBasisInput.text
        if gr_basis != '':
            if not self.grouped_basis_ok():
                return 1

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
                            traceback.print_exc()
                            msg = "All constraints need to be float"
                            App.get_running_app().root.ids.status.text = msg
                            return 1

        return 0

    def get_constraint_matrices(self):
        if self.check_user_input() != 0:
            raise ValueError("Could not parse elements or constraints")

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

    def _clear_constraints(self, clear_elemHeader=True):
        for child in self.ids.mainConcLayout.children[:]:
            if child.id is None:
                continue
            if child.id.startswith('cnst'):
                self.ids.mainConcLayout.remove_widget(child)
            elif child.id == 'elemHeader' and clear_elemHeader:
                self.ids.mainConcLayout.remove_widget(child)

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
                row.append(sign * float(child.text))
                ids.append(child.id)
            elif child.id == 'rhs':
                rhs = sign * float(child.text)

        zipped = sorted(zip(ids, row))
        row = [z[1] for z in zipped]
        return row, rhs, equality

    def apply_Elements_GroupedBasis(self):
        elem_str = self.ids.elementInput.text
        grouped_basis = self.ids.groupedBasisInput.text
        try:
            if grouped_basis != '':
                self.grouped_basis = \
                    parse_grouped_basis_elements(grouped_basis)

            elements = parse_elements(elem_str)
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
            return
        new_elements = self._group_elements(elements, self.grouped_basis)

        if self._elements_changed(new_elements):
            self._clear_constraints()

            self.grouped_elements = new_elements
            self.elements = elements

            layout = StackLayout(id='elemHeader', size_hint=[1, 0.10])
            width = 1.0 / float(self.num_conc_vars + 3)
            layout.add_widget(Label(text='', size_hint=[1, 0.5]))

            for item in self.grouped_elements:
                for sym in item:
                    layout.add_widget(Label(text=sym, size_hint=[width, 0.5]))

            layout.add_widget(Label(text='Type', size_hint=[width, 0.5]))
            layout.add_widget(Label(text='RHS', size_hint=[width, 0.5]))
            layout.add_widget(Label(text='', size_hint=[width, 0.5]))
            self.ids.mainConcLayout.add_widget(layout)

        msg = "Applied Elements and Grouped basis fields."
        App.get_running_app().root.ids.status.text = msg

    def to_dict(self):
        A_lb, rhs_lb, A_eq, rhs_eq = self.get_constraint_matrices()
        data = {
            'elements': self.ids.elementInput.text,
            'grouped_basis': self.ids.groupedBasisInput.text,
            'A_lb': A_lb,
            'rhs_lb': rhs_lb,
            'A_eq': A_eq,
            'rhs_eq': rhs_eq
        }
        return data

    def set_Elements_GroupedBasis(self, elements, grouped_basis):
        self.ids.elementInput.text = elements
        self.ids.groupedBasisInput.text = grouped_basis
        self.apply_Elements_GroupedBasis()

    def load_from_matrices(self, A_lb, rhs_lb, A_eq, rhs_eq):
        # remove constraints if there are any
        self._clear_constraints(clear_elemHeader=False)

        for i in range(len(A_lb)):
            layout = self.add_constraint()

            # Traverse the children
            for child in layout.children:
                if child.id is None:
                    continue
                if child.id.startswith('conc'):
                    col = int(child.id[-1])
                    child.text = str(abs(A_lb[i][col]))
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

    def elem_ok(self):
        elems = self.ids.elementInput.text
        try:
            _ = parse_elements(elems)
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
            return False
        return True

    def grouped_basis_ok(self):
        gr_basis = self.ids.groupedBasisInput.text
        try:
            _ = parse_grouped_basis_elements(gr_basis)
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
            return False
        return True
