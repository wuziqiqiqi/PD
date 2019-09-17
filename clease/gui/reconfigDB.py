from kivy.app import App


class ReconfigDB(object):
    """Reconfigure settings and correlation functions stored in DB."""
    reconfig_in_progress = False
    settings = None
    sm = App.get_running_app().root.ids.sm
    status = App.get_running_app().root.ids.status

    def apply_settings(self):
        if self.reconfig_in_progress:
            # Do no allow user to initialize many threads
            return
        self.reconfig_in_progress = True

        self.status.text = "Reconfiguring in progress..."

        if not self.sm.get_screen('Settings').apply_settings():
            msg = "Couldn't initialize settings based on the values specified "
            msg += "in the fields on Concentration\nand Settings panels. "
            msg += "Please ensure that all of the values are correct."
            self.status.text = msg
            return False
        return True

    def reconfig_settings(self):
        if self.apply_settings():
            print('hello')
            self.settings.reconfigure_settings()
            msg = "Cluster data updated for all templates.\nPlease also "
            msg += "reconfigure DB entries if there are any structures stored "
            msg += "in DB."
            self.status.text = msg
