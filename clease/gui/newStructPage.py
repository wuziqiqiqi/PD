from kivy.uix.screenmanager import Screen
from kivy.utils import get_color_from_hex
from kivy.uix.popup import Popup
from kivy.app import App

from clease.gui.constants import FOREGROUND_TEXT_COLOR, INACTIVE_TEXT_COLOR
from clease.gui.load_save_dialog import LoadDialog
import os
import json
from threading import Thread


class BaseGenerator(object):
    atoms = None
    generator = None
    status = None
    page = None

    def generate(self):
        if self.status is None:
            self.status.text = 'Generate function needs to be implemented'


class RandomStructureGenerator(BaseGenerator):
    def generate(self):
        try:
            self.generator.generate_random_structures(atoms=self.atoms)
            self.status.text = 'Finished generating random structures...'
        except Exception as exc:
            self.status.text = str(exc)
        self.page.structure_generation_in_progress = False


class ProbeStructureGenerator(BaseGenerator):
    Tmin = None
    Tmax = None
    num_temp = None
    num_steps = None

    def generate(self):
        try:
            self.generator.generate_probe_structure(
                atoms=self.atoms, init_temp=self.Tmax, final_temp=self.Tmin,
                num_temp=self.num_temp, num_steps_per_temp=self.num_steps)
            self.status.text = 'Finished generating probe strcutres...'
        except Exception as exc:
            self.status.text = str(exc)
        self.page.structure_generation_in_progress = False


class EminStructGenerator(object):
    Tmax = None
    Tmin = None
    num_temps = None
    num_steps = None
    eci = None
    randomize = None

    def generate(self):
        try:
            self.generator.generate_gs_structure(
                atoms=self.atoms, init_temp=self.Tmax, final_temp=self.Tmin,
                num_temp=self.num_temps, num_steps_per_temp=self.num_steps,
                eci=self.eci, random_composition=self.randomize)
            self.status.text = 'Finished generating GS structures...'
        except Exception as exc:
            self.status.text = str(exc)
        self.page.structure_generation_in_progress = False


class NewStructPage(Screen):
    _pop_up = None
    structure_generation_in_progress = False

    def on_enter(self):
        self.on_new_struct_type_update(self.ids.newStructTypeSpinner.text)

    def on_new_struct_type_update(self, text):
        inactive = get_color_from_hex(INACTIVE_TEXT_COLOR)
        active = get_color_from_hex(FOREGROUND_TEXT_COLOR)

        if text == 'Random structure':
            self.ids.tempMaxLabel.color = inactive
            self.ids.tempMinLabel.color = inactive
            self.ids.numTempLabel.color = inactive
            self.ids.numSweepsLabel.color = inactive
            self.ids.eciFileLabel.color = inactive

            self.ids.tempMaxInput.disabled = True
            self.ids.tempMinInput.disabled = True
            self.ids.numTempInput.disabled = True
            self.ids.numSweepsInput.disabled = True
            self.ids.eciFileInput.disabled = True
            self.ids.randomizeCompositionSpinner.disabled = True
            self.ids.loadECIFile.disabled = True
        elif text == 'Probe structure':
            self.ids.tempMaxLabel.color = active
            self.ids.tempMinLabel.color = active
            self.ids.numTempLabel.color = active
            self.ids.numSweepsLabel.color = active
            self.ids.eciFileLabel.color = inactive

            self.ids.tempMaxInput.disabled = False
            self.ids.tempMinInput.disabled = False
            self.ids.numTempInput.disabled = False
            self.ids.numSweepsInput.disabled = False
            self.ids.eciFileInput.disabled = True
            self.ids.randomizeCompositionSpinner.disabled = True
            self.ids.loadECIFile.disabled = True
        elif text == 'Minimum energy structure':
            self.ids.tempMaxLabel.color = active
            self.ids.tempMinLabel.color = active
            self.ids.numTempLabel.color = active
            self.ids.numSweepsLabel.color = active
            self.ids.eciFileLabel.color = active

            self.ids.tempMaxInput.disabled = False
            self.ids.tempMinInput.disabled = False
            self.ids.numTempInput.disabled = False
            self.ids.numSweepsInput.disabled = False
            self.ids.eciFileInput.disabled = False
            self.ids.randomizeCompositionSpinner.disabled = False
            self.ids.loadECIFile.disabled = False

    def dismiss_popup(self):
        self._pop_up.dismiss()
        self._pop_up = None

    def show_load_eci_dialog(self):
        content = LoadDialog(
            load=lambda path, filename: self.load(path, filename, 'eci'),
            cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load ECI filename", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_load_template_dialog(self):
        content = LoadDialog(
            load=lambda path, filename: self.load(path, filename, 'template'),
            cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load template atoms", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def load(self, path, filename, field='eci'):
        if field == 'eci':
            self.ids.eciFileInput.text = filename[0]
        elif field == 'template':
            self.ids.templateAtomsInput.text = filename[0]
        self.dismiss_popup()

    def load_eci(self, fname):
        with open(fname, 'r') as infile:
            eci = json.load(infile)
        return eci

    def generate(self):
        from clease import NewStructures
        from ase.io import read

        if self.structure_generation_in_progress:
            # Don't allow user to initialise many threads
            # by successively clicking on the generate button
            return

        self.structure_generation_in_progress = True
        settings = App.get_running_app().settings

        if settings is None:
            msg = 'Settings is not set. Make sure Update settings was clicked.'
            self.ids.status.text = msg
            return

        try:
            generator = NewStructures(settings)

            fname = self.ids.templateAtomsInput.text
            if fname == '':
                msg = 'No atoms template given. Using active template.'
                self.ids.status.text = msg
                atoms = settings.atoms.copy()
            else:
                if not os.path.exists(fname):
                    self.ids.status.text = "Cannot find file {}".format(fname)
                    return

                atoms = read(fname)

            struct_type = self.ids.newStructTypeSpinner.text

            Tmin = float(self.ids.tempMinInput.text)
            Tmax = float(self.ids.tempMaxInput.text)
            num_temps = int(self.ids.numTempInput.text)
            num_steps = int(self.ids.numSweepsInput.text)*len(atoms)

            if struct_type == 'Random structure':
                self.ids.status.text = "Generating random structures..."
                rnd_generator = RandomStructureGenerator()
                rnd_generator.generator = generator
                rnd_generator.atoms = atoms
                rnd_generator.status = self.ids.status
                rnd_generator.page = self
                Thread(target=rnd_generator.generate).start()
            elif struct_type == 'Probe structure':
                self.ids.status.text = 'Generating probe structures...'
                prb_generator = ProbeStructureGenerator()
                prb_generator.atoms = atoms
                prb_generator.generator = generator
                prb_generator.status = self.ids.status
                prb_generator.Tmax = Tmax
                prb_generator.Tmin = Tmin
                prb_generator.num_temp = num_temps
                prb_generator.num_steps = num_steps
                prb_generator.page = self
                Thread(target=prb_generator.generate).start()
            elif struct_type == 'Minimum energy structure':
                eci_file = self.ids.eciFileInput.text
                eci = self.load_eci(eci_file)
                self.ids.status.text = 'Generating minimum energy structures.'
                random_comp = self.ids.randomizeCompositionSpinner.text
                randomize = random_comp == 'Random composition'

                emin_generator = EminStructGenerator()
                emin_generator.atoms = atoms
                emin_generator.generator = generator
                emin_generator.status = self.ids.status
                emin_generator.Tmax = Tmax
                emin_generator.Tmin = Tmin
                emin_generator.num_temps = num_temps
                emin_generator.num_steps = num_steps
                emin_generator.eci = eci
                emin_generator.randomize = randomize
                emin_generator.page = self

                Thread(target=emin_generator.generate).start()
        except RuntimeError as exc:
            self.ids.status.text = str(exc)

    def import_structures(self):
        from ase.io import read
        from clease import NewStructures
        init = self.ids.initStructInput.text

        if init == '' or not os.path.exists(init):
            msg = 'Cannot find initial structure {}'.format(init)
            self.ids.status.text = msg
            return

        try:
            init_struct = read(init)

            final = self.ids.finalStructInput.text

            if final == '':
                final_struct = None
            else:
                if not os.path.exists(final):
                    msg = 'Cannot find final structure {}'.format(final)
                    self.ids.status.text = msg
                    return
                final_struct = read(final)

            settings = App.get_running_app().settings

            if settings is None:
                msg = 'Settings is not set. '
                msg += 'Make sure Update settings was clicked.'
                self.ids.status.text = msg
                return
            generator = NewStructures(settings)
            generator.insert_structure(init_struct=init_struct,
                                       final_struct=final_struct,
                                       generate_template=False)
        except Exception as exc:
            self.ids.status.text = str(exc)

    def load_structures(self, path, filename, is_init):
        if len(filename) == 0:
            self.ids.status.text = 'No file selecton...'
            self.dismiss_popup()
            return

        if is_init:
            self.ids.initStructInput.text = filename[0]
        else:
            self.ids.finalStructInput.text = filename[0]
        self.dismiss_popup()

    def show_load_init_struct_dialog(self):
        content = LoadDialog(
            load=lambda path, filename: self.load_structures(path, filename,
                                                             True),
            cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load initial structure", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_load_final_struct_dialog(self):
        content = LoadDialog(
            load=lambda path, filename: self.load_structures(path, filename,
                                                             False),
            cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load final structure", content=content,
                             pos_hint={'right': 0.95, 'top': 0.95},
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def to_dict(self):
        return {
            'init_struct': self.ids.initStructInput.text,
            'final_struct': self.ids.finalStructInput.text,
            'min_temp': self.ids.tempMinInput.text,
            'max_temp': self.ids.tempMaxInput.text,
            'num_temps': self.ids.numTempInput.text,
            'num_sweeps': self.ids.numSweepsInput.text,
            'eci_file': self.ids.eciFileInput.text,
            'template_file': self.ids.templateAtomsInput.text
        }

    def from_dict(self, data):
        self.ids.initStructInput.text = data.get('init_struct', '')
        self.ids.finalStructInput.text = data.get('final_struct', '')
        self.ids.tempMinInput.text = data.get('min_temp', '1')
        self.ids.tempMaxInput.text = data.get('max_temp', '10000')
        self.ids.numTempInput.text = data.get('num_temps', '100')
        self.ids.numSweepsInput.text = data.get('num_sweeps', '100')
        self.ids.eciFileInput.text = data.get('eci_file', '')
        self.ids.templateAtomsInput.text = data.get('template_file', '')
