from clease.gui.cleaseGUI import CleaseGUI
import unittest
import time
from kivy.clock import Clock
from functools import partial


class TestInputPage(unittest.TestCase):
    framecount = 0
    interval = 0.0001

    def pause(*args):
        time.sleep(self.interval)

    def run_test_naviation(self, app):
        Clock.schedule_interval(self.pause, self.interval)

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
        Clock.schedule_interval(self.pause, self.interval)

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
        Clock.schedule_interval(self.pause, self.interval)

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

    def run_tests(self, app):
        Clock.schedule_interval(self.pause, self.interval)
        self.run_test_naviation(app)
        self.run_max_cluster_dia_input(app)
        self.run_cell_size_ok(app)
        app.stop()

    def test_gui(self):
        app = CleaseGUI()
        Clock.schedule_once(lambda x: self.run_tests(app),
                            self.interval)
        app.run()

if __name__ == '__main__':
    unittest.main()
