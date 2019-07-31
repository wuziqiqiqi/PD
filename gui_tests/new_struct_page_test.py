import unittest


class NewStructPageTest(unittest.TestCase):
    def load_pop_ups(self, app):
        screen = app.screen_manager.get_screen('NewStruct')

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadInitStruct.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertEqual(screen._pop_up.title, "Load initial structure")
        screen.dismiss_popup()

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadFinalStruct.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertEqual(screen._pop_up.title, "Load final structure")
        screen.dismiss_popup()

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadECIFile.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertEqual(screen._pop_up.title, "Load ECI filename")
        screen.dismiss_popup()

        self.assertTrue(screen._pop_up is None)
        screen.ids.loadTemplateAtoms.dispatch('on_release')
        self.assertFalse(screen._pop_up is None)
        self.assertEqual(screen._pop_up.title, "Load template atoms")
        screen.dismiss_popup()

    def run(self, app):
        self.load_pop_ups(app)
