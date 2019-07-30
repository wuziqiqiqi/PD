from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.lang import Builder

from inputPage import InputPage
from concentrationPage import ConcentrationPage
from newStructPage import NewStructPage
from fitPage import FitPage

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
