import unittest
from unittest.mock import MagicMock
from clease.gui.fittingAlgorithmEditors import GAEditor
from clease.gui.fitPage import FitPage
import clease.gui
from kivy.uix.textinput import TextInput
from kivy.lang import Builder
from kivy.resources import resource_add_path
import os.path as op
import os

CACHE_FILE = '.cleaseGUI/ga_editor.txt'


def close(ga_edit):
    ga_edit._pop_up.content.ids.closeButton.dispatch('on_release')


def clear_cached_values():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)


class TestGAEditor(unittest.TestCase):
    def test_help_messages(self):
        ga_edit = GAEditor()

        # Help elitism
        ga_edit.ids.helpElitism.dispatch('on_release')
        close(ga_edit)

        # Num individual help
        ga_edit.ids.numIndHelp.dispatch('on_release')
        close(ga_edit)

        # Max active help
        ga_edit.ids.maxActiveHelp.dispatch('on_release')
        close(ga_edit)

        # Cost func help
        ga_edit.ids.costFuncHelp.dispatch('on_release')
        close(ga_edit)

        # Sparsity help
        ga_edit.ids.sparsityHelp.dispatch('on_release')
        close(ga_edit)

        # Force subcluster help
        ga_edit.ids.forceSubHelp.dispatch('on_release')
        close(ga_edit)

    def test_closing(self):
        clear_cached_values()
        page = FitPage()

        page.close_ga_editor = MagicMock()

        # Change algorithm
        page.ids.fitAlgSpinner.text = 'Genetic Algorithm'

        # Launch the fit alg editor
        page.ids.fitEditorButton.dispatch('on_release')

        # It should be possible to close the popup by hitting
        # enter from all text fields
        ga_edit = page._pop_up.content
        num_calls = 0
        for field in ga_edit.ids.mainLayout.children:
            if isinstance(field, TextInput):
                field.dispatch('on_text_validate')
                page.close_ga_editor.assert_called_with(
                    '1', '0.1', '100', '150', 'AIC', '1.0', 'No', False, '100')
                num_calls += 1
                page.close_ga_editor.reset_mock()
        page.dismiss_popup()
        self.assertEqual(num_calls, 6)

if __name__ == '__main__':
    main_path = op.abspath(clease.gui.__file__)
    main_path = main_path.rpartition("/")[0]
    resource_add_path(main_path + '/layout')
    Builder.load_file("gaEditor.kv")
    Builder.load_file("fitLayout.kv")
    Builder.load_file('help_message_layout.kv')
    unittest.main()