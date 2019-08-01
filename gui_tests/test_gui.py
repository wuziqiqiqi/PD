from clease.gui.cleaseGUI import CleaseGUI
import unittest
import time
from kivy.clock import Clock
from functools import partial
from gui_tests.input_page_tests import InputPageTests
from gui_tests.concentraton_page_tests import ConcentrationPageTest
from gui_tests.new_struct_page_test import NewStructPageTest
from gui_tests.fit_page_tests import FitPageTests


class TestInputPage(unittest.TestCase):
    framecount = 0
    interval = 0.0001

    def pause(*args):
        time.sleep(self.interval)

    def run_tests(self, app):
        """
        Main function for running all tests. Note that the application
        is never restarted. Therefore, actions follow each other
        """
        Clock.schedule_interval(self.pause, self.interval)

        input_page_tests = InputPageTests()
        input_page_tests.run(app)

        conc_page_test = ConcentrationPageTest()
        conc_page_test.run(app)

        new_struct_test = NewStructPageTest()
        new_struct_test.run(app)

        fit_page_test = FitPageTests()
        fit_page_test.run(app)
        app.stop()

    def test_gui(self):
        app = CleaseGUI()
        Clock.schedule_once(lambda x: self.run_tests(app),
                            self.interval)
        app.run()

if __name__ == '__main__':
    unittest.main()
