"""Module for generating new structures for training."""
import os
import numpy as np
from random import shuffle
from copy import deepcopy
from functools import reduce

from ase.io import read
from ase.db import connect
from ase.atoms import Atoms
from ase.utils.structure_comparator import SymmetryEquivalenceCheck

from clease import CEBulk, CECrystal, CorrFunction
from clease.tools import wrap_and_sort_by_position, nested_list2str
from clease.structure_generator import ProbeStructure, GSStructure
from clease import _logger
from clease import ValidConcentrationFilter
from itertools import filterfalse

try:
    from math import gcd
except ImportError:
    from fractions import gcd

max_attempt = 10
max_fail = 10


class MaxAttemptReachedError(Exception):
    """Raised when number of try reaches 10."""

    pass


# class GenerateStructures(object):
class NewStructures(object):
    """Generate new structure in Atoms object format.

    Parameters:

    setting: CEBulk or CESpacegroup object

    generation_number: int
        a generation number to be assigned to the newly generated structure.

    struct_per_gen: int
        number of structures to generate per generation.
    """

    def __init__(self, setting, generation_number=None, struct_per_gen=5):
        if not isinstance(setting, (CEBulk, CECrystal)):
            msg = "setting must be CEBulk or CECrystal object"
            raise TypeError(msg)
        self.setting = setting
        self.db = connect(setting.db_name)
        self.corrfunc = CorrFunction(self.setting)
        self.struct_per_gen = struct_per_gen

        if generation_number is None:
            self.gen = self._determine_gen_number()
        else:
            self.gen = generation_number

    def num_in_gen(self):
        return len([row.id for row in self.db.select(gen=self.gen)])

    def num_to_gen(self):
        return max(self.struct_per_gen - self.num_in_gen(), 0)

    def generate_probe_structure(self, atoms=None, size=None,
                                 init_temp=None,
                                 final_temp=None, num_temp=5,
                                 num_steps_per_temp=1000,
                                 approx_mean_var=True,
                                 num_samples_var=10000):
        """Generate a probe structure according to PRB 80, 165122 (2009).

        Parameters:

        atoms: Atoms object
            Atoms object with the desired cell size and shape of the new
            structure.

        size: list of length 3
            (ignored if atoms is given)
            If specified, the structure with the provided size is generated.
            If None, the size will be generated randomly with a bias towards
                more cubic cells (i.e., cell with similar magnitudes of vectors
                a, b and c)

        init_temp: int or float
            initial temperature (does not represent *physical* temperature)

        final_temp: int or float
            final temperature (does not represent *physical* temperature)

        num_temp: int
            number of temperatures to be used in simulated annealing

        num_steps_per_temp: int
            number of steps in simulated annealing

        approx_mean_var: bool
            whether or not to use a spherical and isotropical distribution
            approximation scheme for determining the mean variance.
            -'True': Assume a spherical and isotropical distribution of
                     structures in the configurational space.
                     Corresponds to eq.4 in PRB 80, 165122 (2009)
            -'False': Use sigma and mu of eq.3 in PRB 80, 165122 (2009)
                      to characterize the distribution of structures in
                      population.
                      Requires pre-sampling of random structures before
                      generating probe structures.
                      sigma and mu are generated and stored in
                      'probe_structure-sigma_mu.npz' file.

        num_samples_var: int
            Number of samples to be used in determining signam and mu.
            Only used when approx_mean_var is True.

        Note: init_temp and final_temp are automatically generated if either
              one of the two is not specified.
        """

        if not approx_mean_var:
            # check to see if there are files containing mu and sigma values
            if not os.path.isfile('probe_structure-sigma_mu.npz'):
                self._generate_sigma_mu(num_samples_var)

        _logger("Generate {} probe structures (generation: {}, struct_per_gen={}, {} present)."
                .format(self.num_to_gen(), self.gen, self.struct_per_gen, self.num_in_gen()))

        current_count = 0
        num_attempt = 0
        num_to_generate = self.num_to_gen()

        while current_count < num_to_generate:
            if atoms is not None:
                self.setting.set_active_template(atoms=atoms,
                                                 generate_template=True)
            else:
                self.setting.set_active_template(size=size,
                                                 generate_template=True)
            # Break out of the loop if reached struct_per_gen
            num_struct = self.num_in_gen()
            if num_struct >= self.struct_per_gen:
                break

            struct = self._get_struct_at_conc(conc_type='random')

            _logger('Generating structure {} out of {}.'
                    .format(current_count + 1, num_to_generate))
            ps = ProbeStructure(self.setting, struct, num_to_generate,
                                init_temp, final_temp, num_temp,
                                num_steps_per_temp, approx_mean_var)
            probe_struct, cf = ps.generate()

            # Remove energy from result dictionary
            probe_struct.get_calculator().results.pop('energy')
            formula_unit = self._get_formula_unit(probe_struct)
            if self._exists_in_db(probe_struct, formula_unit):
                msg = 'generated structure is already in DB.\n'
                msg += 'generating again... '
                msg += '{} out of {} attempts'.format(num_attempt + 1,
                                                      max_attempt)
                _logger(msg)
                num_attempt += 1
                if num_attempt >= max_attempt:
                    raise MaxAttemptReachedError("Could not generate probe "
                                                 "structure in {} attempts."
                                                 .format(max_attempt))
            else:
                num_attempt = 0

            _logger('Probe structure generated.\n')
            self.insert_structure(init_struct=probe_struct)
            current_count += 1

    @property
    def corr_func_table_name(self):
        return "{}_cf".format(self.setting.bf_scheme.name)

    def generate_gs_structure_multiple_templates(
            self, num_templates=20, num_prim_cells=2, init_temp=2000,
            final_temp=1, num_temp=10, num_steps_per_temp=1000,
            eci=None):
        """
        Search for ground states in many templates.

        Parameters:

        num_templates: int
            Number of templates to search in. Simmulated annealing is done in
            each cell and the one with the lowest energy is taken as the ground
            state.

        num_prim_cells: int
            Number of primitive cells to use when constructing templates. The
            volume of all the templates used will be
            num_prim_cells*vol_primitive, where vol_primitive is the volume of
            the primitive cell.

        See doc-string of `generate_gs_structure` for the rest of the
        argument.
        """

        for i in range(self.num_to_gen()):
            _logger('Generating ground-state structures: {} of {}'
                    ''.format(i, self.struct_per_gen))

            self._generate_one_gs_structure_multiple_templates(
                num_templates=num_templates, num_prim_cells=num_prim_cells,
                init_temp=init_temp, final_temp=final_temp,
                num_temp=num_temp, num_steps_per_temp=num_steps_per_temp,
                eci=eci
            )

    def _generate_one_gs_structure_multiple_templates(
            self, num_templates=20, num_prim_cells=2, init_temp=2000,
            final_temp=1, num_temp=10, num_steps_per_temp=1000,
            eci=None):
        """
        Search for ground states in many templates.

        Parameters:

        num_templates: int
            Number of templates to search in. Simmulated annealing is done in
            each cell and the one with the lowest energy is taken as the ground
            state.

        num_prim_cells: int
            Number of primitive cells to use when constructing templates. The
            volume of all the templates used will be
            num_prim_cells*vol_primitive, where vol_primitive is the volume of
            the primitive cell.

        See doc-string of `generate_gs_structure` for the rest of the
        argument.
        """
        f = ValidConcentrationFilter(self.setting)
        self.setting.template_atoms.add_atoms_filter(f)
        templates = self.setting.template_atoms.get_fixed_volume_templates(
            num_templates=num_templates, num_prim_cells=num_prim_cells)
        self.setting.template_atoms.remove_filter(f)

        if len(templates) == 0:
            raise RuntimeError("Could not find any templates with matching the "
                               "constraints")

        self.setting.set_active_template(
            atoms=templates[0], generate_template=True)

        nib = [len(x) for x in self.setting.index_by_basis]
        x = self.setting.concentration.get_random_concentration(nib=nib)
        num_insert = self.setting.concentration.conc_in_int(nib, x)

        energies = []
        gs_structs = []
        for i, atoms in enumerate(templates):
            _logger('Searching for GS in template {} of {}'
                    ''.format(i, len(templates)))
            self.setting.set_active_template(
                atoms=atoms, generate_template=True)

            struct = self._random_struct_at_conc(num_insert)
            es = GSStructure(self.setting, struct, self.struct_per_gen,
                             init_temp, final_temp, num_temp,
                             num_steps_per_temp, eci)

            gs_struct, _ = es.generate()
            gs_structs.append(gs_struct)
            energies.append(gs_struct.get_potential_energy())

        # Find the position of the minimum energy structure
        min_energy_indx = np.argmin(energies)
        gs = gs_structs[min_energy_indx]
        self.setting.set_active_template(atoms=gs)
        self.insert_structure(init_struct=gs_structs[min_energy_indx])

    def generate_gs_structure(self, atoms=None, init_temp=2000,
                              final_temp=1, num_temp=10,
                              num_steps_per_temp=1000, eci=None,
                              random_composition=False):
        """Generate ground-state structure.

        Parameters:

        atoms: Atoms object or a list of Atoms object
            Atoms object with the desired size and composition of the new
            structure. A list of Atoms with different size and/or compositions
            can be passed. Compositions of the supplied Atoms object(s) are
            ignored when random_composition=True.

        init_temp: int or float
            initial temperature (does not represent *physical* temperature)

        final_temp: int or float
            final temperature (does not represent *physical* temperature)

        num_temp: int
            number of temperatures to be used in simulated annealing

        num_steps_per_temp: int
            number of steps in simulated annealing

        eci: dict
            Dictionary containing cluster names and their ECI values

        random_composition: bool
            -*False* and atoms = Atoms object: One ground-state structure with
                matching size and composition of the supplied Atoms object is
                generated
            -*False* and atoms = list: The same number of ground-state
                structures that matches the length of the list is generated
                Note 1: num_struct_per_gen is ignored and all of the generated
                        structures have the same generation number
                Note 2: each GS structure will have matching size and
                        composition of the suplied Atoms objects
            -*True* and atoms = Atoms object: GS structure(s) with a
                matching size of the Atoms object is generated at a random
                composition (within the composition range specified in
                Concentration class)
                Note 1: This will generate GS structures until the number of
                        structures with the current generation number equals
                        num_struct_per_gen
                Note 2: A check is performed to ensure that none of the newly
                        generated GS structures have the same composition
            -*True* and atoms = list: The same number of GS structures that
                matches the length of the list is generated
                Note 1: num_struct_per_gen is ignored and all of the generated
                        structures have the same generation number
                Note 2: each GS structure will have matching sizes of the
                        supplied Atoms objects but with a random composition
                Note 3: No check is performed to ensure that all new GS
                        structures have unique composition
        """
        structs = self._set_initial_structures(atoms, random_composition)
        current_count = 0
        num_attempt = 0
        num_to_generate = min([self.num_to_gen(), len(structs)])
        while current_count < num_to_generate:
            struct = structs[current_count].copy()
            self.setting.set_active_template(atoms=struct,
                                             generate_template=False)
            _logger("Generating structure {} out of {}."
                    .format(current_count + 1, num_to_generate))
            es = GSStructure(self.setting, struct, self.struct_per_gen,
                             init_temp, final_temp, num_temp,
                             num_steps_per_temp, eci)
            gs_struct, cf = es.generate()
            formula_unit = self._get_formula_unit(gs_struct)

            if self._exists_in_db(gs_struct, formula_unit):
                msg = 'generated structure is already in DB.\n'
                msg += 'generating again... '
                msg += '{} out of {} attempts'.format(num_attempt + 1,
                                                      max_attempt)
                _logger(msg)
                num_attempt += 1
                if num_attempt >= max_attempt:
                    raise MaxAttemptReachedError(
                        "Could not generate ground-state structure in "
                        "{} attempts.".format(max_attempt))
                continue
            else:
                num_attempt = 0

            min_energy = gs_struct.get_potential_energy()
            msg = 'Structure with E = {:.3f} generated.\n'.format(min_energy)
            _logger(msg)
            self.insert_structure(init_struct=gs_struct)
            current_count += 1

    def generate_random_structures(self, atoms=None):
        """Generate a given number of random structures.

        Parameters:

        atoms: Atoms object or None.
            If Atoms object is passed, the passed object will be used as a
            template for all the random structures being generated.
            If None, a random template will be chosen
            (different for each structure)
        """
        _logger("Generating {} random structures "
                "(generation: {}, struct_per_gen={}, {} present)"
                .format(self.num_to_gen(), self.gen, self.struct_per_gen,
                        self.num_in_gen()))

        fail_counter = 0
        i = 0

        num_structs = self.num_to_gen()
        while i < num_structs and fail_counter < max_fail:
            if self.generate_one_random_structure(atoms=atoms, verbose=False):
                _logger("Generated {} random structures".format(i + 1))
                i += 1
                fail_counter = 0
            else:
                fail_counter += 1

        if fail_counter >= max_fail:
            RuntimeError("Could not find a structure that does not exist in "
                         "DB after {} attempts."
                         "".format(int(max_attempt * max_fail)))

    def generate_one_random_structure(self, atoms=None, verbose=True):
        """Generate a random structure.

        Inserts a new structure to database if a unique structure is found.

        Returns ``True`` if unique structure is found and inserted in DB,
        ``False`` otherwise.

        Parameters:

        atoms: Atoms object or None.
            If Atoms object is passed, the passed object will be used as a
            template for all the random structures being generated.
            If None, a random template will be chosen
            (different for each structure)

        verbose: bool
            If ``True``, print error message when it reaches max. number of
            attempts.
        """
        num_attempts = 0
        unique_structure_found = False

        while not unique_structure_found and num_attempts < max_attempt:
            self.setting.set_active_template(atoms=atoms,
                                             generate_template=False)
            new_atoms = self._get_struct_at_conc(conc_type="random")
            fu = self._get_formula_unit(new_atoms)
            if not self._exists_in_db(new_atoms, fu):
                unique_structure_found = True
            else:
                num_attempts += 1

        if not unique_structure_found:
            if verbose:
                _logger("Could not find a structure that does not already "
                        "exist in the DB within {} attempts."
                        "".format(max_attempt))
            return False

        self.insert_structure(init_struct=new_atoms)
        return True

    def _set_initial_structures(self, atoms, random_composition=False):
        structs = []
        if isinstance(atoms, Atoms):
            struct = wrap_and_sort_by_position(atoms)
            if random_composition is False:
                num_to_gen = 1
                _logger("Generate 1 ground-state structure.")
                structs.append(struct)
            else:
                _logger("Generate {} ground-state structures "
                        "(generation: {}, struct_per_gen={}, {} present)"
                        .format(self.num_to_gen(),
                                self.gen,
                                self.struct_per_gen,
                                self.num_in_gen()))
                self.setting.set_active_template(atoms=struct,
                                                 generate_template=True)
                num_to_gen = self.num_to_gen()
                concs = []
                # Get unique concentrations
                num_attempt = 0
                nib = [len(x) for x in self.setting.index_by_basis]
                while len(concs) < num_to_gen:
                    x = self.setting.concentration.get_random_concentration(
                        nib=nib)
                    if True in [np.allclose(x, i) for i in concs]:
                        num_attempt += 1
                    else:
                        concs.append(x)
                        num_attempt = 0

                    if num_attempt > 100:
                        raise RuntimeError("Could not find {} unique "
                                           "compositions using the provided "
                                           "Atoms object"
                                           .format(self.num_to_gen))
                num_atoms_in_basis = [len(indices) for indices
                                      in self.setting.index_by_basis]
                for x in concs:
                    num_insert = self.setting.concentration.conc_in_int(
                        num_atoms_in_basis, x)
                    structs.append(self._random_struct_at_conc(num_insert))

        elif all(isinstance(a, Atoms) for a in atoms):
            _logger("Generate {} ground-state structures ".format(len(atoms)))
            if random_composition is False:
                for struct in atoms:
                    structs.append(wrap_and_sort_by_position(struct))
            else:
                concs = []
                nib = [len(x) for x in self.setting.index_by_basis]
                for struct in atoms:
                    self.setting.set_active_template(atoms=struct,
                                                     generate_template=True)
                    x = self.setting.concentration.get_random_concentration(
                        nib=nib)
                    num_atoms_in_basis = [len(indices) for indices
                                          in self.setting.index_by_basis]
                    num_insert = self.setting.concentration.conc_in_int(
                        num_atoms_in_basis, x)
                    structs.append(self._random_struct_at_conc(num_insert))

        else:
            raise ValueError("atoms must be either an Atoms object or a list "
                             "of Atoms objects")
        return structs

    def generate_initial_pool(self, atoms=None):
        """Generate initial pool of structures.

        Initial pool of structures are generated, in sequence, using

        1. generate_conc_extrema(): structures at concentration where the
           number of consituting elements is at its max/min.
        2. generate_random_structures(): random structures are random
           concentration.

        Parameters:

        atoms: Atoms object | None.
            If Atoms object is passed, the passed object will be used as a
            template for all the random structures being generated.
            If None, a random template will be chosen
            (different for each structure)."""

        self.generate_conc_extrema()
        self.generate_random_structures(atoms=atoms)

    def generate_conc_extrema(self):
        """Generate initial pool of structures with max/min concentration."""
        from itertools import product
        _logger("Generating one structure per concentration where the number "
                "of an element is at max/min")
        indx_in_each_basis = []
        start = 0
        for basis in self.setting.concentration.basis_elements:
            indx_in_each_basis.append(list(range(start, start + len(basis))))
            start += len(basis)

        for indx in product(*indx_in_each_basis):
            atoms = self._get_struct_at_conc(conc_type="max", index=indx)
            atoms = wrap_and_sort_by_position(atoms)
            self.insert_structure(init_struct=atoms)

    def _get_struct_at_conc(self, conc_type='random', index=0):
        """Generate a structure at a concentration specified.

        Parameters:

        conc_type: str
            One of 'min', 'max' and 'random'

        index: int
            index of the flattened basis_element array to specify which
            element to be maximized/minimized
        """
        conc = self.setting.concentration
        if conc_type == 'min':
            x = conc.get_conc_min_component(index)
        elif conc_type == 'max':
            x = conc.get_conc_max_component(index)
        else:
            nib = [len(x) for x in self.setting.index_by_basis]
            x = conc.get_random_concentration(nib=nib)

        num_atoms_in_basis = [len(indices) for indices
                              in self.setting.index_by_basis]
        num_atoms_to_insert = conc.conc_in_int(num_atoms_in_basis, x)
        atoms = self._random_struct_at_conc(num_atoms_to_insert)

        return atoms

    def insert_structures(self, traj_init=None, traj_final=None):
        """
        Insert a sequence of initial and final structures from trajectory.

        Parameters:

        traj_init: str
            Filename of a trajectory file with initial structures

        traj_final: str
            Filename of a trajectory file with the final structures
        """
        from ase.io.trajectory import TrajectoryReader
        from clease.tools import count_atoms
        traj_in = TrajectoryReader(traj_init)
        traj_final = TrajectoryReader(traj_final)

        if len(traj_in) != len(traj_final):
            raise ValueError("Different number of structures in "
                             "initial trajectory file and final.")

        for init, final in zip(traj_in, traj_final):
            # Check that composition (except vacancies matches)
            count_init = count_atoms(init)
            count_final = count_atoms(final)
            for k in count_final.keys():
                if k not in count_init.keys():
                    raise ValueError("Final and initial structure contains "
                                     "different elements")

                if count_init[k] != count_final[k]:
                    raise ValueError("Final and initial structure has "
                                     "different number of each species")

            self.insert_structure(init_struct=init, final_struct=final,
                                  generate_template=True)

    def insert_structure(self, init_struct=None, final_struct=None, name=None,
                         generate_template=False):
        """Insert a user-supplied structure to the database.

        Parameters:

        init_struct: .xyz, .cif or .traj file
            Unrelaxed initial structure.

        final_struct: .traj file (optional)
            Final structure that contains the energy.
            Needs to also supply init_struct in order to use the final_struct.

        name: str (optional)
            Name of the DB entry if a custom name is to be used.
            If ``None``, default naming convention will be used.

        generate_template: bool (optional)
            If set to ``True``, a template matching the size of the passed
            ``init_struct`` is created in DB.
        """
        if init_struct is None:
            raise TypeError('init_struct must be provided')

        if name is not None:
            num = sum(1 for _ in self.db.select(['name', '=', name]))
            if num > 0:
                raise ValueError("Name: {} already exists in DB!".format(name))

        if isinstance(init_struct, Atoms):
            init = wrap_and_sort_by_position(init_struct)
        else:
            init = wrap_and_sort_by_position(read(init_struct))

        self.setting.set_active_template(atoms=init_struct,
                                         generate_template=generate_template)

        formula_unit = self._get_formula_unit(init)
        if self._exists_in_db(init, formula_unit):
            _logger('Supplied structure already exists in DB.'
                    'The structure will not be inserted.')
            return

        cf = self.corrfunc.get_cf(init)
        kvp = self._get_kvp(init, formula_unit)

        if name is not None:
            kvp['name'] = name

        kvp['converged'] = False
        kvp['started'] = False
        kvp['queued'] = False
        kvp['struct_type'] = 'initial'
        tab_name = self.corr_func_table_name
        uid_init = self.db.write(init, kvp, external_tables={tab_name: cf})

        if final_struct is not None:
            if isinstance(final_struct, Atoms):
                final = final_struct
            else:
                final = read(final_struct)
            kvp_final = {'struct_type': 'final', 'name': kvp['name']}
            uid = self.db.write(final, kvp_final)
            self.db.update(uid_init, converged=True, started='', queued='',
                           final_struct_id=uid)

    def _exists_in_db(self, atoms, formula_unit=None):
        """Check to see if the passed atoms already exists in DB.

        To reduce the number of assessments for symmetry equivalence,
        check is only performed with the entries with the same concentration
        value.

        Return *True* if there is a symmetry-equivalent structure in DB,
        return *False* otherwise.

        Parameters:

        atoms: Atoms object

        formula_unit: str
            reduced formula unit of the passed Atoms object
        """
        cond = [('name', '!=', 'template'), ('name', '!=', 'primitive_cell')]
        if formula_unit is not None:
            cond.append(("formula_unit", "=", formula_unit))

        to_prim = True
        try:
            __import__('spglib')
        except Exception:
            msg = "Warning! Setting to_primitive=False because spglib "
            msg += "is missing!"
            _logger(msg)
            to_prim = False

        symmcheck = SymmetryEquivalenceCheck(angle_tol=1.0, ltol=0.05,
                                             stol=0.05, scale_volume=True,
                                             to_primitive=to_prim)
        atoms_in_db = []
        for row in self.db.select(cond):
            atoms_in_db.append(row.toatoms())
        return symmcheck.compare(atoms.copy(), atoms_in_db)

    def _get_kvp(self, atoms, formula_unit=None):
        """Get key-value pairs of the passed Atoms object.

        Create key-value pairs and return it.

        Parameters:

        atoms: Atoms object

        formula_unit: str
            reduced formula unit of the passed Atoms object
        """
        if formula_unit is None:
            raise ValueError("Formula unit not specified!")
        kvp = {}
        kvp['gen'] = self.gen
        kvp['converged'] = False
        kvp['started'] = False
        kvp['queued'] = False

        count = 0
        for _ in self.db.select(formula_unit=formula_unit):
            count += 1
        kvp['name'] = formula_unit + "_{}".format(count)
        kvp['formula_unit'] = formula_unit
        kvp['struct_type'] = 'initial'
        kvp['size'] = nested_list2str(self.setting.size)
        return kvp

    def _get_formula_unit(self, atoms):
        """Generates a reduced formula unit for the structure."""
        atom_count = []
        all_nums = []
        for group in self.setting.index_by_basis:
            new_count = {}
            for indx in group:
                symbol = atoms[indx].symbol
                if symbol not in new_count.keys():
                    new_count[symbol] = 1
                else:
                    new_count[symbol] += 1
            atom_count.append(new_count)
            all_nums += [v for k, v in new_count.items()]
        gcdp = reduce(lambda x, y: gcd(x, y), all_nums)
        fu = ""
        for i, count in enumerate(atom_count):
            keys = list(count.keys())
            keys.sort()
            for k in keys:
                fu += "{}{}".format(k, int(count[k] / gcdp))
            if i < len(atom_count) - 1:
                fu += "_"
        return fu

    def _random_struct_at_conc(self, num_atoms_to_insert):
        """Generate a random structure."""
        rnd_indices = []
        for indices in self.setting.index_by_basis:
            rnd_indices.append(deepcopy(indices))
            shuffle(rnd_indices[-1])

        # Insert the number of atoms
        basis_elem = self.setting.concentration.basis_elements
        assert len(rnd_indices) == len(basis_elem)
        atoms = self.setting.atoms.copy()
        current_conc = 0
        num_atoms_inserted = 0
        for basis in range(len(rnd_indices)):
            current_indx = 0
            for symb in basis_elem[basis]:
                for _ in range(num_atoms_to_insert[current_conc]):
                    atoms[rnd_indices[basis][current_indx]].symbol = symb
                    current_indx += 1
                    num_atoms_inserted += 1
                current_conc += 1
        assert num_atoms_inserted == len(atoms)
        return atoms

    def _determine_gen_number(self):
        """Determine generation number based on the values in DB."""
        try:
            gens = [row.get('gen') for row in self.db.select()]
            gens = [i for i in gens if i is not None]
            gen = max(gens)
            num_in_gen = len([row.id for row in self.db.select(gen=gen)])
            if num_in_gen >= self.struct_per_gen:
                gen += 1
        except ValueError:
            gen = 0
        return gen

    def _generate_sigma_mu(self, num_samples_var):
        _logger('===========================================================\n'
                'Determining sigma and mu value for assessing mean variance.\n'
                'May take a long time depending on the number of samples \n'
                'specified in the *num_samples_var* argument.\n'
                '===========================================================')
        count = 0
        cfm = np.zeros((num_samples_var, len(self.setting.all_cf_names)),
                       dtype=float)
        while count < num_samples_var:
            atoms = self._get_struct_at_conc(conc_type='random')
            cfm[count] = self.corrfunc.get_cf(atoms, 'array')
            count += 1
            _logger('sampling {} ouf of {}'.format(count, num_samples_var))

        sigma = np.cov(cfm.T)
        mu = np.mean(cfm, axis=0)
        np.savez('probe_structure-sigma_mu.npz', sigma=sigma, mu=mu)
