from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty
from clease.gui.mc_main_page import MCMainPage
from clease.gui.canonical_mc_page import CanonicalMCPage
from kivy.app import App
from threading import Thread
from clease.gui.constants import SCREEN_TRANSLATIONS


class MCHeader(Screen):
    mc_type = StringProperty('Main')

    def change_screen(self, txt):
        all_mc_screens = self.ids.pageSpinner.values

        cur_screen = self.ids.sm.current
        cur_key = None
        for k, v in SCREEN_TRANSLATIONS.items():
            if v == cur_screen:
                cur_key = k
                break

        i_cur = all_mc_screens.index(cur_key)
        i_next = all_mc_screens.index(txt)

        if i_next > i_cur:
            self.ids.sm.transition.direction = 'left'
        else:
            self.ids.sm.transition.direction = 'right'
        self.ids.sm.current = SCREEN_TRANSLATIONS[txt]

    @property
    def mc_cell_size(self):
        page = self.ids.sm.get_screen('MCMainPage')
        return int(page.ids.sizeInput.text)
