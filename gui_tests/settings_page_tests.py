import unittest


class SettingsPageTests(unittest.TestCase):
    def run_test_naviation(self, app):
        # Make sure that we are on the SettingsPage
        self.assertEqual('Concentration', app.root.ids.sm.current)

        screens = ['Concentration', 'Settings', 'NewStruct', 'Fit']
        for main_screen in screens:
            screen = app.root.ids.sm.get_screen(main_screen)

            screen.ids.toSettings.dispatch('on_release')
            self.assertEqual('Settings', app.root.ids.sm.current)

            screen.ids.concEditor.dispatch('on_release')

            screen.ids.toNewStruct.dispatch('on_release')
            self.assertEqual('NewStruct', app.root.ids.sm.current)

            screen.ids.toFit.dispatch('on_release')
            self.assertEqual('Fit', app.root.ids.sm.current)

    def run_max_cluster_dia_input(self, app):
        screen = app.root.ids.sm.get_screen('Settings')

        # Set maximum cluster size to 4
        screen.ids.clusterSize.text = '4'

        # Try invalid string
        screen.ids.clusterDia.text = 'adfadf'
        self.assertFalse(screen.max_cluster_dia_ok())

        # Try a float number
        screen.ids.clusterDia.text = '5.0'
        self.assertTrue(screen.max_cluster_dia_ok())

        # Try int number
        screen.ids.clusterDia.text = '4'
        self.assertTrue(screen.max_cluster_dia_ok())

        # Try list with wrong size
        screen.ids.clusterDia.text = '7.0, 4'
        self.assertFalse(screen.max_cluster_dia_ok())

        # Try list with correct size
        screen.ids.clusterDia.text = '7.0, 5.0, 6.0'
        self.assertTrue(screen.max_cluster_dia_ok())

    def run_cell_size_ok(self, app):
        screen = app.root.ids.sm.get_screen('Settings')
        screen.ids.sizeInput.text = 'df'
        self.assertFalse(screen.cell_size_ok())

        screen.ids.sizeInput.text = '3, 3'
        self.assertFalse(screen.cell_size_ok())

        screen.ids.sizeInput.text = '3, 3, 4'
        self.assertTrue(screen.cell_size_ok())

        screen.ids.sizeInput.text = '[[0, 1, 0], [1, 0, 1], [-1, 2, 0]]'
        self.assertFalse(screen.cell_size_ok())

        screen.ids.sizeInput.text = '[(0, 1, 0), (1, 0, 1), (-1, 2, 0)]'
        self.assertFalse(screen.cell_size_ok())

        screen.ids.sizeInput.text = '((0, 1, 0), (1, 0, 1), (-1, 2, 0))'
        self.assertTrue(screen.cell_size_ok())

        screen.ids.sizeInput.text = '(0, 1, 0), (1, 0, 1), (-1, 2, 0)'
        self.assertTrue(screen.cell_size_ok())

    def run_load_dialog(self, app):
        screen = app.root.ids.sm.get_screen('Settings')
        screen.ids.loadDbButton.dispatch('on_release')
        self.assertEqual(screen._pop_up.title, 'Load structure DB')
        screen._pop_up.content.ids.cancelButton.dispatch('on_release')

    def run_check_cellpar(self, app):
        screen = app.root.ids.sm.get_screen('Settings')

        screen.ids.cellParInput.text = 'dx'
        self.assertFalse(screen.cellpar_ok())

        screen.ids.cellParInput.text = '(3.0, 4.0)'
        self.assertFalse(screen.cellpar_ok())

        screen.ids.cellParInput.text = '6.0, 7.0, 3.0, 80, 20, 10'
        self.assertTrue(screen.cellpar_ok())

        screen.ids.cellParInput.text = '6.0, 7.0, 3.0, 80, 20'
        self.assertFalse(screen.cellpar_ok())

        screen.ids.cellParInput.text = '(6.0, 7.0, 3.0, 80, 20, 10)'
        self.assertFalse(screen.cellpar_ok())

    def run_check_cell_input(self, app):
        screen = app.root.ids.sm.get_screen('Settings')
        cell_inp = screen.ids.cellInput

        cell_inp.text = 'db'
        self.assertFalse(screen.cell_ok())

        cell_inp.text = '(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)'
        self.assertTrue(screen.cell_ok())

    def run_check_element_input(self, app):
        screen = app.root.ids.sm.get_screen('Concentration')
        elem_in = screen.ids.elementInput

        screen = app.root.ids.sm.get_screen('Settings')
        elem_in.text = 'Al, Cu'
        self.assertTrue(screen.elem_ok())

        elem_in.text = '(Al, Cu), (Mg, Si)'
        self.assertTrue(screen.elem_ok())

    def run_check_grouped_basis(self, app):
        screen = app.root.ids.sm.get_screen('Concentration')
        gr_basis = screen.ids.groupedBasisInput
        screen = app.root.ids.sm.get_screen('Settings')

        gr_basis.text = '1, 2'
        self.assertTrue(screen.grouped_basis_ok())

        gr_basis.text = '(1, 2), 3'
        self.assertTrue(screen.grouped_basis_ok())

        gr_basis = '(1, 2), (3, 4)'
        self.assertTrue(screen.grouped_basis_ok())

    def run_save_as_button(self, app):
        screen = app.root.ids.sm.get_screen('Settings')
        conc_screen = app.root.ids.sm.get_screen('Concentration')
        self.assertTrue(screen._pop_up is None)

        # Call save button with an incomplete page
        screen.ids.saveAsSession.dispatch('on_release')
        self.assertTrue(screen._pop_up is None)

        # Populate the fields with a valid input
        screen.ids.aParameterInput.text = '4.05'
        screen.ids.dbNameInput.text = 'test_gui.db'
        conc_screen.ids.elementInput.text = 'Au, Cu'
        screen.ids.saveAsSession.dispatch('on_release')
        self.assertTrue(screen._pop_up is not None)
        self.assertEqual(screen._pop_up.title, "Save CLEASE session")
        screen.dismiss_popup()

    def run_load_session_button(self, app):
        screen = app.root.ids.sm.get_screen('Settings')

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadSession.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertTrue(screen._pop_up.title, "Load CLEASE session")
        screen.dismiss_popup()

    def run(self, app):
        self.run_test_naviation(app)
        self.run_max_cluster_dia_input(app)
        self.run_cell_size_ok(app)
        self.run_load_dialog(app)
        self.run_check_cellpar(app)
        self.run_check_cell_input(app)
        self.run_check_element_input(app)
        self.run_check_grouped_basis(app)
        self.run_save_as_button(app)
        self.run_load_session_button(app)
