from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.lang import Builder

from clease.gui.inputPage import InputPage
from clease.gui.concentrationPage import ConcentrationPage
from clease.gui.newStructPage import NewStructPage
from clease.gui.fitPage import FitPage
from kivy.resources import resource_add_path
import os.path as op

main_path = op.abspath(__file__)
main_path = main_path.rpartition("/")[0]
resource_add_path(main_path + '/layout')

Builder.load_file("cleaseGUILayout.kv")


class CleaseGUI(App):
    def __init__(self):
        App.__init__(self)
        self.screen_manager = ScreenManager()
        self.screen_manager.add_widget(InputPage(name="Input"))
        self.screen_manager.add_widget(ConcentrationPage(name="Concentration"))
        self.screen_manager.add_widget(NewStructPage(name='NewStruct'))
        self.screen_manager.add_widget(FitPage(name='Fit'))
        self.screen_manager.current = "Input"
        self.settings = None

    def build(self):
        return self.screen_manager


if __name__ == "__main__":
    CleaseGUI().run()
