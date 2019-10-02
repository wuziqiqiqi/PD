import unittest


class ConcentrationPageTest(unittest.TestCase):

    def check_add_constraint(self, app):
        screen = app.root.ids.sm.get_screen('Concentration')
        num_children = len(screen.ids.mainConcLayout.children)
        screen.ids.addConstraintButton.dispatch('on_release')
        self.assertEqual(len(screen.ids.mainConcLayout.children),
                         num_children + 1)

        # Locate the remove button
        for widget in screen.ids.mainConcLayout.children:
            if widget.id is None:
                continue
            if widget.id == 'cnst0':
                for ch in widget.children:
                    if ch.text == 'Remove':
                        ch.dispatch('on_press')
        self.assertEqual(len(screen.ids.mainConcLayout.children),
                         num_children)

    def check_matrices(self, app):
        screen = app.root.ids.sm.get_screen('Concentration')
        screen.ids.elementInput.text = 'Al, Mg, Si'
        screen.ids.groupedBasisInput.text = ''
        screen.ids.applyElemGroupButton.dispatch('on_release')

        # Constraint counter is on 1 after the check_add_constraint
        self.assertEqual(screen.next_constraint_id, 1)
        screen.ids.addConstraintButton.dispatch('on_release')
        screen.ids.addConstraintButton.dispatch('on_release')
        screen.ids.addConstraintButton.dispatch('on_release')
        self.assertEqual(screen.next_constraint_id, 4)

        for widget in screen.ids.mainConcLayout.children:
            if widget.id is None:
                continue

            if widget.id == 'cnst1':
                for ch in widget.children:
                    if ch.id is None:
                        continue

                    if ch.id == 'conc0' or ch.id == 'conc1':
                        ch.text = '1.0'
                    elif ch.id == 'rhs':
                        ch.text = '0.5'
                    elif ch.id == 'comparisonSpinner':
                        ch.text = '='
            elif widget.id == 'cnst2':
                for ch in widget.children:
                    if ch.id is None:
                        continue

                    if ch.id == 'conc2':
                        ch.text = '1.0'
                    elif ch.id == 'rhs':
                        ch.text = '0.5'
                    elif ch.id == 'comparisonSpinner':
                        ch.text = '<='
            elif widget.id == 'cnst3':
                for ch in widget.children:
                    if ch.id is None:
                        continue

                    if ch.id == 'conc1':
                        ch.text = '1.0'
                    elif ch.id == 'rhs':
                        ch.text = '0.5'
                    elif ch.id == 'comparisonSpinner':
                        ch.text = '>='
        A_lb, rhs_lb, A_eq, rhs_eq = screen.get_constraint_matrices()

        A_lb_expect = [[0, 1, 0], [0, 0, -1]]
        rhs_lb_expect = [0.5, -0.5]
        A_eq_expect = [[1, 1, 0]]
        rhs_eq_expect = [0.5]

        for row, row_exp in zip(A_lb, A_lb_expect):
            for x, y in zip(row, row_exp):
                self.assertAlmostEqual(x, y)

        for row, row_exp in zip(A_eq, A_eq_expect):
            for x, y in zip(row, row_exp):
                self.assertAlmostEqual(x, y)

        for x, y in zip(rhs_lb, rhs_lb_expect):
            self.assertAlmostEqual(x, y)

        for x, y in zip(rhs_eq, rhs_eq_expect):
            self.assertAlmostEqual(x, y)

    def run_check_element_input(self, app):
        screen = app.root.ids.sm.get_screen('Concentration')
        elem_in = screen.ids.elementInput
        elem_in.text = 'Al, Cu'
        self.assertTrue(screen.elem_ok())
        elem_in.text = '[[Zn, Cu]]'
        self.assertTrue(screen.elem_ok())
        elem_in.text = '(Al, Cu), (Mg, Si)'
        self.assertTrue(screen.elem_ok())
        elem_in.text = '[(Al, Cu), (Mg, Si)]'
        self.assertTrue(screen.elem_ok())

    def run_check_grouped_basis(self, app):
        screen = app.root.ids.sm.get_screen('Concentration')
        gr_basis = screen.ids.groupedBasisInput
        gr_basis.text = '1, 2'
        self.assertTrue(screen.grouped_basis_ok())
        gr_basis.text = '[1, 2]'
        self.assertTrue(screen.grouped_basis_ok())
        gr_basis.text = '[1, 2], 3'
        self.assertTrue(screen.grouped_basis_ok())
        gr_basis.text = '(1, 2), [3, 4]'
        self.assertTrue(screen.grouped_basis_ok())
        gr_basis.text = '(1, 2), (3, 4)'
        self.assertTrue(screen.grouped_basis_ok())
        gr_basis.text = '[(1, 2), (3, 4)]'
        self.assertTrue(screen.grouped_basis_ok())

    def run_with_app(self, app):
        self.check_add_constraint(app)
        self.check_matrices(app)
        self.run_check_element_input(app)
        self.run_check_grouped_basis(app)
