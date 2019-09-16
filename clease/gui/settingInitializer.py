from kivy.app import App
import traceback


class SettingsInitializer(object):
    """Perform settings initialization on a separate thread."""
    type = 'CEBulk'
    kwargs = None
    app = None
    status = None

    def initialize(self):
        from clease import CEBulk, CECrystal
        try:
            if self.type == 'CEBulk':
                # self.app.settings = CEBulk(**self.kwargs)
                App.get_running_app().root.settings = CEBulk(**self.kwargs)
            elif self.type == 'CECrystal':
                # self.app.settings = CECrystal(**self.kwargs)
                App.get_running_app().root.settings = CECrystal(**self.kwargs)
            msg = "Database initialized"
            App.get_running_app().root.ids.status.text = msg

        except AssertionError as exc:
            traceback.print_exc()
            msg = "AssertError during initialization " + str(exc)
            App.get_running_app().root.ids.status.text = msg

        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)