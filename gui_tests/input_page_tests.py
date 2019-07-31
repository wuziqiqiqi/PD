import unittest


class InputPageTests(unittest.TestCase):
    def run_test_naviation(self, app):
        # Make sure that we are on the InputPage
        self.assertEqual('Input', app.screen_manager.current)

        screens = ['Input', 'Concentration', 'NewStruct', 'Fit']
        for main_screen in screens:
            screen = app.screen_manager.get_screen(main_screen)

            screen.ids.toInput.dispatch('on_release')
            self.assertEqual('Input', app.screen_manager.current)

            screen.ids.concEditor.dispatch('on_release')

            screen.ids.toNewStruct.dispatch('on_release')
            self.assertEqual('NewStruct', app.screen_manager.current)

            screen.ids.toFit.dispatch('on_release')
            self.assertEqual('Fit', app.screen_manager.current)

    def run_max_cluster_dia_input(self, app):
        screen = app.screen_manager.get_screen('Input')

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
        screen = app.screen_manager.get_screen('Input')
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
        screen = app.screen_manager.get_screen('Input')
        screen.ids.loadDbButton.dispatch('on_release')
        self.assertEqual(screen._pop_up.title, 'Load structure DB')
        screen._pop_up.content.ids.cancelButton.dispatch('on_release')

    def run_check_cellpar(self, app):
        screen = app.screen_manager.get_screen('Input')

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
        screen = app.screen_manager.get_screen('Input')
        cell_inp = screen.ids.cellInput

        cell_inp.text = 'db'
        self.assertFalse(screen.cell_ok())

        cell_inp.text = '(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)'
        self.assertTrue(screen.cell_ok())

    def run_check_element_input(self, app):
        screen = app.screen_manager.get_screen('Input')
        elem_in = screen.ids.elementInput

        elem_in.text = 'Al, Cu'
        self.assertTrue(screen.elem_ok())

        elem_in.text = '(Al, Cu), (Mg, Si)'
        self.assertTrue(screen.elem_ok())

    def run_check_grouped_basis(self, app):
        screen = app.screen_manager.get_screen('Input')

        gr_basis = screen.ids.groupedBasisInput
        gr_basis.text = '1, 2'
        self.assertTrue(screen.grouped_basis_ok())

        gr_basis.text = '(1, 2), 3'
        self.assertTrue(screen.grouped_basis_ok())

        gr_basis = '(1, 2), (3, 4)'
        self.assertTrue(screen.grouped_basis_ok())

    def run_save_as_button(self, app):
        screen = app.screen_manager.get_screen('Input')
        self.assertTrue(screen._pop_up is None)

        # Call save button with an incomplete page
        screen.ids.saveAsSession.dispatch('on_release')
        self.assertTrue(screen._pop_up is None)

        # Populate the fields with a valid input
        screen.ids.aParameterInput.text = '4.05'
        screen.ids.dbNameInput.text = 'test_gui.db'
        screen.ids.elementInput.text = 'Au, Cu'
        screen.ids.saveAsSession.dispatch('on_release')
        self.assertTrue(screen._pop_up is not None)
        self.assertEqual(screen._pop_up.title, "Save CLEASE session")
        screen.dismiss_popup()

    def run_load_session_button(self, app):
        screen = app.screen_manager.get_screen('Input')

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
