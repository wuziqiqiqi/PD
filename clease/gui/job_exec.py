from kivy.uix.screenmanager import Screen
from clease.gui.load_save_dialog import LoadDialog
from clease.gui.help_message_popup import HelpMessagePopup
from kivy.uix.popup import Popup
from kivy.app import App
import subprocess
from threading import Thread
from clease.gui.constants import JOB_EXEC_MSG


class JobExec(Screen):
    _pop_up = None

    def open_file_dialog(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._pop_up = Popup(title="Load script",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def to_dict(self):
        """Return content as dict."""
        return {
            'cmd': self.ids.cmdInput.text,
            'script': self.ids.scriptInput.text,
            'dbIds': self.ids.dbIdsInput.text,
            'additionCmdArgs': self.ids.cmdArgsInput.text
        }

    def from_dict(self, dct):
        self.ids.cmdInput.text = dct.get('cmd', 'python')
        self.ids.scriptInput.text = dct.get('script', 'myscript.py')
        self.ids.dbIdsInput.text = dct.get('dbIds', '')
        self.ids.cmdArgsInput.text = dct.get('additionCmdArgs')

    def dismiss_popup(self):
        if self._pop_up is None:
            return
        self._pop_up.dismiss()
        self._pop_up = None

    def load(self, path, filename):
        self.db_path = path

        if len(filename) == 0:
            self.ids.scriptInput.text = path
        else:
            self.ids.scriptInput.text = filename[0]
        self.dismiss_popup()

    def show_help_msg(self):
        msg = 'The program will execute the following command:\n'
        msg += '<cmd> <script> dbID <arg1> <arg2> <arg3>...,\n'
        msg += 'where <cmd> and <script> are the two first entries on\n'
        msg += 'the page. dbID is an integer and <arg1>, <arg2> etc.\n'
        msg += 'are optional arguments given as a comma separated list\n'
        msg += 'If you give a comma separated list of IDs, the command \n'
        msg += 'will be launched for each ID. Furthermore, you can also\n'
        msg += 'specify a range of IDs like this 4-7 which is expanded to\n'
        msg += '4, 5, 6, 7.'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Calculation help",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_args_help(self):
        msg = 'In many cases, the IDs of the calculations are not sufficient\n'
        msg += 'and require more arguments to be passed. One example is that\n'
        msg += 'you want to perform an initial relaxation with lower energy\n'
        msg += 'cutoff and k-point density, followed by another calculation\n'
        msg += 'with higher energy cutoff and k-point density. You can \n'
        msg += 'specify additional arguments by passing comma-separated\n'
        msg += 'values. For instance, if your script can accept energy cutoff\n'
        msg += 'and k-point density as arguments, you can add 500, 5.4 to the\n'
        msg += 'field to pass the respective values. The executed command\n'
        msg += 'when the run button is pressed will be:\n\n'
        msg += 'python myscript.py <dBId> 500 5.4\n'
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title="Additional args help",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def run_on_separate_thread(self, cmd, script, ids, args):
        app = App.get_running_app()
        for i, uid in enumerate(ids):
            msg = f"Running job {i} of {len(ids)}"
            app.root.ids.status.text = msg
            subprocess.check_call([cmd, script, str(uid)] + args)
        app.root.ids.status.text = JOB_EXEC_MSG['jobs_finished']

    def _resolve_id_ranges(self, dbIds):
        """Resolve ranges IDs."""
        ids = []
        for item in dbIds:
            if '-' in item:
                splitted = item.split('-')
                min_id = int(splitted[0])
                max_id = int(splitted[1])
                ids += list(range(min_id, max_id + 1))
            else:
                ids.append(int(item))
        return ids

    def run(self):
        args = self.ids.cmdArgsInput.text.split(',')
        app = App.get_running_app()
        try:
            dbIds = self.ids.dbIdsInput.text.split(',')
            dbIds = self._resolve_id_ranges(dbIds)
        except Exception:
            app.root.ids.status.text = JOB_EXEC_MSG['db_id_error']
            return

        cmd = self.ids.cmdInput.text
        script = self.ids.scriptInput.text
        Thread(target=self.run_on_separate_thread, args=(cmd, script, dbIds, args)).start()
