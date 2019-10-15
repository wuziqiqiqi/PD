from kivy.app import App
import traceback


class ReconfigDB(object):
    """Reconfigure settings and correlation functions stored in DB."""
    # reconfig_in_progress = False
    settings = None
    app = None
    status = None

    def apply_settings(self):

        if not self.app.root.ids.sm.get_screen('Settings').apply_settings():
            msg = "Couldn't initialize settings based on the values specified "
            msg += "in the fields on Concentration\nand Settings panels. "
            msg += "Please ensure that all of the values are correct."
            self.status.text = msg
            return False

        self.status.text = "Reconfiguring..."
        return True

    def reconfig_settings(self):
        if self.apply_settings():
            try:
                self.app.root.settings.reconfigure_settings()
                msg = "Cluster data updated for all templates.\nPlease also "
                msg += "reconfigure DB entries if there are any structures "
                msg += "stored in DB."
                self.status.text = msg

            except Exception as exc:
                traceback.print_exc()
                self.status.text = str(exc)

        self.app.root.reconfig_in_progress = False

    def reconfig_db(self):
        if self.apply_settings():
            try:
                from clease import CorrFunction
                CorrFunction(self.app.root.settings).reconfigure_db_entries()
                msg = "All DB entries are reconfigured."
                self.status.text = msg

            except Exception as exc:
                traceback.print_exc()
                self.status.text = str(exc)

        self.app.root.reconfig_in_progress = False

    def reconfig_settings_db(self):
        if self.apply_settings():
            try:
                from clease.tools import reconfigure
                reconfigure(self.app.root.settings)
                msg = "Settings andall DB entries are reconfigured."
                self.status.text = msg

            except Exception as exc:
                traceback.print_exc()
                self.status.text = str(exc)

        self.app.root.reconfig_in_progress = False
