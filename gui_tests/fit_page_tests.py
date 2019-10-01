import unittest


class FitPageTests(unittest.TestCase):
    def eci_popup(self, app):
        screen = app.root.ids.sm.get_screen('Fit')

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadEciInput.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertEqual(screen._pop_up.title, "Load ECI file")
        screen.dismiss_popup()

    def open_fit_alg_editors(self, app):
        screen = app.root.ids.sm.get_screen('Fit')
        spinner = screen.ids.fitAlgSpinner

        last_pop_title = ""
        for value in spinner.values:
            spinner.text = value
            self.assertTrue(screen._pop_up is None)
            screen.ids.fitEditorButton.dispatch('on_release')
            self.assertFalse(screen._pop_up is None)
            new_title = screen._pop_up.title

            # We don't explicitly check the title, but in order to make
            # sure that the pop up actually changes we check that the
            # new title is different from the previous
            self.assertNotEqual(new_title, last_pop_title)
            screen.dismiss_popup()
            last_pop_title = new_title

    def run_with_app(self, app):
        self.eci_popup(app)
        self.open_fit_alg_editors(app)
