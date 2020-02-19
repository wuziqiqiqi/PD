from kivy.app import App
import traceback


class SettingsInitializer(object):
    """Perform settings initialization on a separate thread."""
    type = 'CEBulk'
    kwargs = None
    basis_func_type = 'polynomial'
    skew_threshold = 40
    app = None
    status = None

    def initialize(self):
        from clease import CEBulk, CECrystal
        try:
            if self.type == 'CEBulk':
                self.app.root.settings = CEBulk(**self.kwargs)
            elif self.type == 'CECrystal':
                self.app.root.settings = CECrystal(**self.kwargs)
            self.status.text = 'Finished initializing'
            self.app.root.settings.basis_func_type = self.basis_func_type
            self.app.root.settings.skew_threshold = self.skew_threshold
        except AssertionError as exc:
            traceback.print_exc()
            msg = "AssertError during initialization " + str(exc)
            self.status.text = msg

        except Exception as exc:
            traceback.print_exc()
            self.status.text = str(exc)
