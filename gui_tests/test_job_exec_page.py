import unittest
from unittest.mock import patch, MagicMock
import clease.gui
from kivy.lang import Builder
from kivy.resources import resource_add_path
from clease.gui.job_exec import JobExec
from kivy.uix.textinput import TextInput
import os.path as op
from textwrap import dedent
import os
import time

class TestJobExecPage(unittest.TestCase):
    @patch('clease.gui.job_exec.App')
    def test_to_from_dict(self, app):
        page = JobExec()
        
        num_text_items = 0
        for item in page.walk():
            if  isinstance(item, TextInput):
                num_text_items += 1
                item.text = 'Some_text'
        
        dct = page.to_dict()
        self.assertEqual(len(dct.keys()), num_text_items)

        page2 = JobExec()
        page2.from_dict(dct)

        for item in page2.children[0].children:
            if isinstance(item, TextInput):
                self.assertEqual(item.text, 'Some_text')

    @patch('clease.gui.job_exec.App')
    def test_load_script(self, app):
        page = JobExec()
        self.assertIsNone(page._pop_up)

        page.ids.loadButton.dispatch('on_release')
        self.assertIsNotNone(page._pop_up)

        page._pop_up.content.ids.loadButton.dispatch('on_release')
        self.assertIsNone(page._pop_up)

    @patch('clease.gui.job_exec.App')
    def test_help_popup(self, app):
        page = JobExec()
        self.assertIsNone(page._pop_up)

        page.ids.helpButton.dispatch('on_release')
        self.assertIsNotNone(page._pop_up)

        page._pop_up.content.ids.closeButton.dispatch('on_release')
        self.assertIsNone(page._pop_up)

    @patch('clease.gui.job_exec.App')
    def test_run(self, app):
        page = JobExec()

        script = """
            import sys

            def main(argv):
                uid = argv[0]
                other = argv[1]
                assert other == 'additional_arg'

            main(sys.argv[1:])
        """

        script_name = 'some_script.py'
        with open(script_name, 'w') as f:
            f.write(dedent(script))
        
        page.ids.scriptInput.text = script_name
        page.ids.cmdArgsInput.text = 'additional_arg'
        page.ids.dbIdsInput.text = '1'

        page.ids.runButton.dispatch('on_release')

        # Try with many IDs
        page.ids.dbIdsInput.text = '1, 2, 5'
        page.ids.runButton.dispatch('on_release')

        # Since the jobs are executing on a separate thread, we need to wait
        # before we delete the script. 0.5s should be more than enough
        time.sleep(0.5)
        os.remove(script_name)

if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("cleaseGUILayout.kv")
    unittest.main()
    unittest.main()
