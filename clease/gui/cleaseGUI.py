from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout

from clease.gui.settingsPage import SettingsPage
from clease.gui.concentrationPage import ConcentrationPage
from clease.gui.newStructPage import NewStructPage
from clease.gui.fitPage import FitPage
from kivy.resources import resource_add_path
import os.path as op

main_path = op.abspath(__file__)
main_path = main_path.rpartition("/")[0]
resource_add_path(main_path + '/layout')

Builder.load_file("cleaseGUILayout.kv")

class WindowFrame(BoxLayout):
    pass

class CleaseGUI(App):
    def __init__(self):
        App.__init__(self)
        self.settings = None

    def build(self):
        return WindowFrame()


if __name__ == "__main__":
    CleaseGUI().run()
