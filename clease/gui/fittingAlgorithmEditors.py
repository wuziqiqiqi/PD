from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from clease.gui.help_message_popup import HelpMessagePopup
import os
from kivy.uix.popup import Popup


class FitAlgEditor(FloatLayout):
    close = ObjectProperty(None)
    backup_file = 'fit_alg_editor.txt'
    backup_folder = '.cleaseGUI/'
    _pop_up = None

    def __init__(self, **kwargs):
        FloatLayout.__init__(self, **kwargs)
        if not os.path.exists(self.backup_folder):
            os.mkdir(self.backup_folder)
        self.load_values()

    @property
    def backup_filename(self):
        return self.backup_folder + self.backup_file

    def backup(self):
        values = []
        for child in self.ids.mainLayout.children:
            if isinstance(child, (TextInput, Spinner)):
                txt = child.text
                values.append(txt)

        with open(self.backup_filename, 'w') as out:
            for v in values:
                out.write('{}\n'.format(v))

    def load_values(self):
        try:
            with open(self.backup_filename, 'r') as infile:
                lines = infile.readlines()
            lines = [line.strip() for line in lines]

            counter = 0
            for child in self.ids.mainLayout.children:
                if isinstance(child, (TextInput, Spinner)):
                    child.text = lines[counter]
                    counter += 1
        except Exception:
            pass

    def dismiss_popup(self):
        self._pop_up.dismiss()
        self._pop_up = None

    def show_help_message(self, msg, title):
        content = HelpMessagePopup(close=self.dismiss_popup, msg=msg)
        self._pop_up = Popup(title=title, content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()


class LassoEditor(FitAlgEditor):
    backup_file = 'lasso_editor.txt'


class L2Editor(FitAlgEditor):
    backup_file = 'l2_editor.txt'


class BCSEditor(FitAlgEditor):
    backup_file = 'bcs_editor.txt'


class GAEditor(FitAlgEditor):
    backup_file = 'ga_editor.txt'

    def show_elitism_help(self):
        msg = 'Number of inidividuals that are transferred\n'
        msg += 'to the next generation, without any pairing.\n'
        msg += 'If this is 1, only the best inidividual is\n'
        msg += 'transferred. If it is 2, the two most fit\n'
        msg += 'inidviduals is transferred'
        self.show_help_message(msg, 'Elitism help')

    def show_sparsity_help(self):
        msg = 'For several cost function a penalisation term\n'
        msg += 'which is a monotonic function of the number of\n'
        msg += 'active clusters in the chosen string of clusters\n'
        msg += 'and the number of datapoint is added. For BIC his\n'
        msg += 'k*ln(N), where N is the number of datapoints and k\n'
        msg += 'is the number of chosen features. We allow a modified\n'
        msg += 'version of this. alpha*k*ln(N), where alphs is the\n'
        msg += 'called sparsity slope. This is only effective for\n'
        msg += 'BIC, AIC, AICC and HQC'
        self.show_help_message(msg, 'Sparsity help')

    def show_individual_help(self):
        msg = 'An individual in the CE context is a binary\n'
        msg += 'representation of clusters. If the clusters\n'
        msg += 'are lined up in an array, the individual can be\n'
        msg += '[0, 0, 0, 1, 1, 0, 1, 0, 1, 1] which means that\n'
        msg += 'cluster 4, 5, 7, 8 and 9 is included in the fit\n'
        msg += 'GA tries to find the individual that optimises\n'
        msg += 'the chosen cost function'
        self.show_help_message(msg, 'Individual help')

    def show_max_active_help(self):
        msg = 'If the pool of clusters is large, it might be wise\n'
        msg += 'to limit the maximum of active clusters in the initial\n'
        msg += 'population. This number specifies the maximum allowed\n'
        msg += 'number of 1 in any individual in the initial pool'
        self.show_help_message(msg, 'Max. active pool')

    def show_cost_func_help(self):
        msg = 'Mathematical symbols:\n'
        msg += 'k - number of selected features\n'
        msg += 'n - number of datapoints\n\n'
        msg += 'Cost functions:\n'
        msg += 'AIC   - Afaike Information Criterion (penalty 2*k)\n'
        msg += 'AICC  - Modified AIC (AICC = AIC + (2k^2 + 2k)/(n-k-1)\n'
        msg += 'BIC   - Bayes Information Criterion (penalty k*ln(n))\n'
        msg += 'LOOCV - Leave-one-out-cross validation\n'
        msg += 'HQS   - Hannan-Quin Criterion (penalty 2*k*ln(ln(n)))\n'
        self.show_help_message(msg, 'Cost function help')

    def show_subcluster_help(self):
        msg = 'If Yes all subclusters of all included clusters have\n'
        msg += 'to be included. Hence, if a 3-body cluster is selected\n'
        msg += 'all the 2-body clusters forming it, also have to be\n'
        msg += 'included.'
        self.show_help_message(msg, 'Force subcluster help')