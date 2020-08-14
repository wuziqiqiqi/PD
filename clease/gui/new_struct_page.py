from kivy.uix.screenmanager import Screen
from kivy.utils import get_color_from_hex
from kivy.uix.popup import Popup
from kivy.app import App

from clease.gui.constants import FOREGROUND_TEXT_COLOR, INACTIVE_TEXT_COLOR
from clease.gui.load_save_dialog import LoadDialog
import os
import json
from threading import Thread
import traceback


class BaseGenerator(object):
    atoms = None
    generator = None
    status = None
    page = None

    def generate(self):
        if self.status is None:
            msg = 'Generate function needs to be implemented.'
            App.get_running_app().root.ids.status.text = msg


class InitialPoolGenerator(BaseGenerator):

    def generate(self):
        try:
            self.generator.generate_initial_pool(atoms=self.atoms)
            msg = 'Initial pool of structures generated.'
            App.get_running_app().root.ids.status.text = msg
        except Exception as exc:
            App.get_running_app().root.ids.status.text = str(exc)
        self.page.structure_generation_in_progress = False


class RandomStructureGenerator(BaseGenerator):

    def generate(self):
        try:
            self.generator.generate_random_structures(atoms=self.atoms)
            msg = 'Random structures generated.'
            App.get_running_app().root.ids.status.text = msg
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
        self.page.structure_generation_in_progress = False


class ProbeStructureGenerator(BaseGenerator):
    Tmin = None
    Tmax = None
    num_temp = None
    num_steps = None

    def generate(self):
        try:
            self.generator.generate_probe_structure(atoms=self.atoms,
                                                    init_temp=self.Tmax,
                                                    final_temp=self.Tmin,
                                                    num_temp=self.num_temp,
                                                    num_steps_per_temp=self.num_steps)
            msg = 'Probe strcutres generated.'
            App.get_running_app().root.ids.status.text = msg
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
        self.page.structure_generation_in_progress = False


class GSStructGenerator(object):
    Tmax = None
    Tmin = None
    num_temps = None
    num_steps = None
    eci = None
    randomize = None

    def generate(self):
        try:
            self.generator.generate_gs_structure(atoms=self.atoms,
                                                 init_temp=self.Tmax,
                                                 final_temp=self.Tmin,
                                                 num_temp=self.num_temps,
                                                 num_steps_per_temp=self.num_steps,
                                                 eci=self.eci,
                                                 random_composition=self.randomize)
            msg = 'Ground-state structures generated.'
            App.get_running_app().root.ids.status.text = msg
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
        self.page.structure_generation_in_progress = False


class InsertStructureCB(object):
    """
    Callable class used to update the status field when inserting
    structures

    :param status: widget
        Status field of the page
    """

    def __init__(self, status):
        self.status = status

    def __call__(self, num, tot):
        self.status.text = f"Inserted {num} of {tot} structures"


class ExaustiveGSStructGenerator(object):
    Tmax = None
    Tmin = None
    num_temps = None
    num_steps = None
    eci = None
    num_templates = None
    num_prim_cells = None

    def generate(self):
        try:
            self.generator.generate_gs_structure_multiple_templates(
                num_templates=self.num_templates,
                num_prim_cells=self.num_prim_cells,
                init_temp=self.Tmax,
                final_temp=self.Tmin,
                num_temp=self.num_temps,
                num_steps_per_temp=self.num_steps,
                eci=self.eci)
            msg = 'Ground-state structures generated.'
            App.get_running_app().root.ids.status.text = msg
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
        self.page.structure_generation_in_progress = False


class NewStructPage(Screen):
    _pop_up = None
    structure_generation_in_progress = False

    def on_enter(self):
        self.on_new_struct_type_update(self.ids.newStructTypeSpinner.text)

    def on_new_struct_type_update(self, text):
        inactive = get_color_from_hex(INACTIVE_TEXT_COLOR)
        active = get_color_from_hex(FOREGROUND_TEXT_COLOR)

        if text in ['Initial pool', 'Random structure']:
            self.ids.templateAtomsLabel.color = active
            self.ids.tempMaxLabel.color = inactive
            self.ids.tempMinLabel.color = inactive
            self.ids.numTempLabel.color = inactive
            self.ids.numSweepsLabel.color = inactive
            self.ids.eciFileLabel.color = inactive
            self.ids.numTemplateLabel.color = inactive
            self.ids.numPrimCells.color = inactive

            self.ids.loadTemplateAtoms.disabled = False
            self.ids.templateAtomsInput.disabled = False
            self.ids.tempMaxInput.disabled = True
            self.ids.tempMinInput.disabled = True
            self.ids.numTempInput.disabled = True
            self.ids.numSweepsInput.disabled = True
            self.ids.eciFileInput.disabled = True
            self.ids.randomizeCompositionSpinner.disabled = True
            self.ids.loadECIFile.disabled = True
            self.ids.numTemplateInput.disabled = True
            self.ids.numPrimCellsInput.disabled = True

        elif text == 'Probe structure':
            self.ids.tempMaxLabel.color = active
            self.ids.tempMinLabel.color = active
            self.ids.numTempLabel.color = active
            self.ids.numSweepsLabel.color = active
            self.ids.templateAtomsLabel.color = active
            self.ids.eciFileLabel.color = inactive
            self.ids.numTemplateLabel.color = inactive
            self.ids.numPrimCells.color = inactive

            self.ids.tempMaxInput.disabled = False
            self.ids.tempMinInput.disabled = False
            self.ids.numTempInput.disabled = False
            self.ids.numSweepsInput.disabled = False
            self.ids.loadTemplateAtoms.disabled = False
            self.ids.templateAtomsInput.disabled = False
            self.ids.eciFileInput.disabled = True
            self.ids.randomizeCompositionSpinner.disabled = True
            self.ids.loadECIFile.disabled = True
            self.ids.numTemplateInput.disabled = True
            self.ids.numPrimCellsInput.disabled = True

        elif text == 'Ground-state structure (fixed template)':
            self.ids.tempMaxLabel.color = active
            self.ids.tempMinLabel.color = active
            self.ids.numTempLabel.color = active
            self.ids.numSweepsLabel.color = active
            self.ids.eciFileLabel.color = active
            self.ids.templateAtomsLabel.color = active
            self.ids.numTemplateLabel.color = inactive
            self.ids.numPrimCells.color = inactive

            self.ids.tempMaxInput.disabled = False
            self.ids.tempMinInput.disabled = False
            self.ids.numTempInput.disabled = False
            self.ids.numSweepsInput.disabled = False
            self.ids.eciFileInput.disabled = False
            self.ids.randomizeCompositionSpinner.disabled = False
            self.ids.loadECIFile.disabled = False
            self.ids.loadTemplateAtoms.disabled = False
            self.ids.templateAtomsInput.disabled = False
            self.ids.numTemplateInput.disabled = True
            self.ids.numPrimCellsInput.disabled = True

        elif text == 'Ground-state structure (variable template)':
            self.ids.tempMaxLabel.color = active
            self.ids.tempMinLabel.color = active
            self.ids.numTempLabel.color = active
            self.ids.numSweepsLabel.color = active
            self.ids.eciFileLabel.color = active
            self.ids.numTemplateLabel.color = active
            self.ids.numPrimCells.color = active
            self.ids.templateAtomsLabel.color = inactive

            self.ids.tempMaxInput.disabled = False
            self.ids.tempMinInput.disabled = False
            self.ids.numTempInput.disabled = False
            self.ids.numSweepsInput.disabled = False
            self.ids.eciFileInput.disabled = False
            self.ids.randomizeCompositionSpinner.disabled = True
            self.ids.loadECIFile.disabled = False
            self.ids.numTemplateInput.disabled = False
            self.ids.numPrimCellsInput.disabled = False
            self.ids.loadTemplateAtoms.disabled = True
            self.ids.templateAtomsInput.disabled = True

    def dismiss_popup(self):
        self._pop_up.dismiss()
        self._pop_up = None

    def show_load_eci_dialog(self):
        content = LoadDialog(load=lambda path, filename: self.load(path, filename, 'eci'),
                             cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load ECI filename",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_load_template_dialog(self):
        content = LoadDialog(load=lambda path, filename: self.load(path, filename, 'template'),
                             cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load template atoms",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
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

    def get_eci_file(self):
        eci_file = self.ids.eciFileInput.text

        if eci_file == '':
            raise ValueError('No ECI file given')
        return eci_file

    def generate(self):
        from clease import NewStructures
        from ase.io import read

        if self.structure_generation_in_progress:
            # Don't allow user to initialize many threads
            # by successively clicking on the generate button
            return

        self.structure_generation_in_progress = True
        settings = App.get_running_app().root.settings

        if settings is None:
            msg = "Settings is not set. "
            msg += "Make sure Apply settings button was clicked.'"
            App.get_running_app().root.ids.status.text = msg
            self.structure_generation_in_progress = False
            return

        if self.ids.genNumberInput.text == '':
            generation_number = None
        else:
            generation_number = int(self.ids.genNumberInput.text)
        struct_per_gen = int(self.ids.structPerGenInput.text)

        try:
            generator = NewStructures(settings,
                                      generation_number=generation_number,
                                      struct_per_gen=struct_per_gen)

            fname = self.ids.templateAtomsInput.text
            if fname == '':
                msg = 'No Atoms template given. Using active template.'
                App.get_running_app().root.ids.status.text = msg
                atoms = settings.atoms.copy()
            else:
                if not os.path.exists(fname):
                    msg = f"Cannot find file {fname}"
                    App.get_running_app().root.ids.status.text = msg
                    self.structure_generation_in_progress = False
                    return

                atoms = read(fname)

            struct_type = self.ids.newStructTypeSpinner.text

            Tmin = float(self.ids.tempMinInput.text)
            Tmax = float(self.ids.tempMaxInput.text)
            num_temps = int(self.ids.numTempInput.text)
            num_steps = int(self.ids.numSweepsInput.text) * len(atoms)
            num_templates = int(self.ids.numTemplateInput.text)
            num_prim_cells = int(self.ids.numPrimCellsInput.text)

            if struct_type == 'Initial pool':
                msg = "Generating initial pool of structures..."
                App.get_running_app().root.ids.status.text = msg
                init_generator = InitialPoolGenerator()
                init_generator.generator = generator
                init_generator.atoms = atoms
                init_generator.status = App.get_running_app().root.ids.status
                init_generator.page = self
                Thread(target=init_generator.generate).start()
            elif struct_type == 'Random structure':
                msg = "Generating random structures..."
                App.get_running_app().root.ids.status.text = msg
                rnd_generator = RandomStructureGenerator()
                rnd_generator.generator = generator
                rnd_generator.atoms = atoms
                rnd_generator.status = App.get_running_app().root.ids.status
                rnd_generator.page = self
                Thread(target=rnd_generator.generate).start()
            elif struct_type == 'Probe structure':
                msg = 'Generating probe structures...'
                App.get_running_app().root.ids.status.text = msg
                prb_generator = ProbeStructureGenerator()
                prb_generator.atoms = atoms
                prb_generator.generator = generator
                prb_generator.status = App.get_running_app().root.ids.status
                prb_generator.Tmax = Tmax
                prb_generator.Tmin = Tmin
                prb_generator.num_temp = num_temps
                prb_generator.num_steps = num_steps
                prb_generator.page = self
                Thread(target=prb_generator.generate).start()
            elif struct_type == 'Ground-state structure (fixed template)':
                eci_file = self.get_eci_file()
                eci = self.load_eci(eci_file)
                msg = 'Generating ground-state structures...'
                App.get_running_app().root.ids.status.text = msg
                random_comp = self.ids.randomizeCompositionSpinner.text
                randomize = random_comp == 'Random composition'

                gs_generator = GSStructGenerator()
                gs_generator.atoms = atoms
                gs_generator.generator = generator
                gs_generator.status = App.get_running_app().root.ids.status
                gs_generator.Tmax = Tmax
                gs_generator.Tmin = Tmin
                gs_generator.num_temps = num_temps
                gs_generator.num_steps = num_steps
                gs_generator.eci = eci
                gs_generator.randomize = randomize
                gs_generator.page = self

                Thread(target=gs_generator.generate).start()
            elif struct_type == 'Ground-state structure (variable template)':
                eci_file = self.get_eci_file()
                eci = self.load_eci(eci_file)
                msg = 'Generating ground-state structures...'
                App.get_running_app().root.ids.status.text = msg

                gs_generator = ExaustiveGSStructGenerator()
                gs_generator.generator = generator
                gs_generator.status = App.get_running_app().root.ids.status
                gs_generator.Tmax = Tmax
                gs_generator.Tmin = Tmin
                gs_generator.num_temps = num_temps
                gs_generator.num_steps = num_steps
                gs_generator.eci = eci
                gs_generator.num_templates = num_templates
                gs_generator.num_prim_cells = num_prim_cells
                gs_generator.page = self

                Thread(target=gs_generator.generate).start()
        except (RuntimeError, ValueError) as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)
            self.structure_generation_in_progress = False

    def import_structures(self):
        from ase.io import read
        from clease import NewStructures
        init = self.ids.initStructInput.text

        if init == '' or not os.path.exists(init):
            msg = f"Cannot find initial structure {init}"
            App.get_running_app().root.ids.status.text = msg
            return

        try:
            init_struct = read(init)

            final = self.ids.finalStructInput.text

            if final == '':
                final_struct = None
            else:
                if not os.path.exists(final):
                    msg = f"Cannot find final structure {final}"
                    App.get_running_app().root.ids.status.text = msg
                    return
                final_struct = read(final)

            settings = App.get_running_app().root.settings

            if settings is None:
                msg = 'Settings is not set. '
                msg += 'Make sure Apply settings button was clicked.'
                App.get_running_app().root.ids.status.text = msg
                return

            if self.ids.genNumberInput.text == '':
                generation_number = None
            else:
                generation_number = int(self.ids.genNumberInput.text)
            struct_per_gen = int(self.ids.structPerGenInput.text)

            generator = NewStructures(settings,
                                      generation_number=generation_number,
                                      struct_per_gen=struct_per_gen)

            status = App.get_running_app().root.ids.status
            status.text = 'Inserting structures...'

            # The argument passed is a trajectory file
            if final.endswith('.traj') and init.endswith('.traj'):
                kwargs = {'traj_init': init, 'traj_final': final, 'cb': InsertStructureCB(status)}
                Thread(target=generator.insert_structures, kwargs=kwargs).start()
            elif init.endswith('.traj'):
                kwargs = {'traj_init': init, 'cb': InsertStructureCB(status)}
                Thread(target=generator.insert_structures, kwargs=kwargs).start()
            else:
                generator.insert_structure(init_struct=init_struct, final_struct=final_struct)
        except Exception as exc:
            traceback.print_exc()
            App.get_running_app().root.ids.status.text = str(exc)

    def load_structures(self, path, filename, is_init):
        if len(filename) == 0:
            App.get_running_app().root.ids.status.text = 'No file selecton...'
            self.dismiss_popup()
            return

        if is_init:
            self.ids.initStructInput.text = filename[0]
        else:
            self.ids.finalStructInput.text = filename[0]
        self.dismiss_popup()

    def show_load_init_struct_dialog(self):
        content = LoadDialog(load=lambda path, filename: self.load_structures(path, filename, True),
                             cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load initial structure",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def show_load_final_struct_dialog(self):
        content = LoadDialog(
            load=lambda path, filename: self.load_structures(path, filename, False),
            cancel=self.dismiss_popup)

        self._pop_up = Popup(title="Load final structure",
                             content=content,
                             pos_hint={
                                 'right': 0.95,
                                 'top': 0.95
                             },
                             size_hint=(0.9, 0.9))
        self._pop_up.open()

    def to_dict(self):
        return {
            'gen_number': self.ids.genNumberInput.text,
            'struct_per_gen': self.ids.structPerGenInput.text,
            'init_struct': self.ids.initStructInput.text,
            'final_struct': self.ids.finalStructInput.text,
            'min_temp': self.ids.tempMinInput.text,
            'max_temp': self.ids.tempMaxInput.text,
            'num_temps': self.ids.numTempInput.text,
            'num_sweeps': self.ids.numSweepsInput.text,
            'eci_file': self.ids.eciFileInput.text,
            'template_file': self.ids.templateAtomsInput.text,
            'generation_scheme': self.ids.newStructTypeSpinner.text,
            'num_templates': self.ids.numTemplateInput.text
        }

    def from_dict(self, data):
        self.ids.genNumberInput.text = data.get('gen_number', '')
        self.ids.structPerGenInput.text = data.get('struct_per_gen', '5')
        self.ids.initStructInput.text = data.get('init_struct', '')
        self.ids.finalStructInput.text = data.get('final_struct', '')
        self.ids.tempMinInput.text = data.get('min_temp', '1')
        self.ids.tempMaxInput.text = data.get('max_temp', '10000')
        self.ids.numTempInput.text = data.get('num_temps', '100')
        self.ids.numSweepsInput.text = data.get('num_sweeps', '100')
        self.ids.eciFileInput.text = data.get('eci_file', '')
        self.ids.templateAtomsInput.text = data.get('template_file', '')
        self.ids.newStructTypeSpinner.text = data.get('generation_scheme', 'Random structure')
        self.ids.numTemplateInput.text = data.get('num_templates', '1')
