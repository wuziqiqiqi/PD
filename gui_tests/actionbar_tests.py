import unittest


class ActionBarTests(unittest.TestCase):
    def run_save_as_button(self, app):
        settings_screen = app.root.ids.sm.get_screen('Settings')
        conc_screen = app.root.ids.sm.get_screen('Concentration')
        self.assertTrue(app.root._pop_up is None)

        # Call save button with an incomplete page
        app.root.show_save_dialog()
        self.assertTrue(app.root._pop_up is None)

        # Populate the fields with a valid input
        conc_screen.ids.elementInput.text = 'Au, Cu'
        conc_screen.ids.groupedBasisInput.text = ''
        conc_screen.ids.applyElemGroupButton.dispatch('on_release')
        settings_screen.ids.aParameterInput.text = '4.05'
        settings_screen.ids.dbNameInput.text = 'test_gui.db'
        settings_screen.ids.sizeInput.text = '3, 3, 3'
        settings_screen.ids.sizeSpinner.text = 'Fixed'
        settings_screen.ids.typeSpinner.text = 'CEBulk'
        app.root.show_save_dialog()
        self.assertFalse(app.root._pop_up is None)
        self.assertEqual(app.root._pop_up.title, "Save CLEASE session")
        app.root.dismiss_popup()

    def run_load_session_button(self, app):
        self.assertTrue(app.root._pop_up is None)
        app.root.show_load_session_dialog()
        self.assertFalse(app.root._pop_up is None)
        self.assertTrue(app.root._pop_up.title, "Load CLEASE session")
        app.root.dismiss_popup()

    def run_with_app(self, app):
        self.run_save_as_button(app)
        self.run_load_session_button(app)