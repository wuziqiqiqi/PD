"""Class containing a manager for creating template atoms."""
import os
import numpy as np
from itertools import product
from random import choice
from ase.db import connect
from ase.build import cut
from itertools import combinations
from ase.build import make_supercell
from clease.tools import str2nested_list
from clease import _logger
from clease import SkewnessFilter, EquivalentCellsFilter
from clease.template_filters import CellFilter, AtomsFilter


class TemplateAtoms(object):
    def __init__(self, supercell_factor=None, size=None, skew_threshold=4,
                 db_name=None):
        if size is None and supercell_factor is None:
            raise TypeError("Either size or supercell_factor needs to be "
                            "specified.\n size: list or numpy array.\n "
                            "supercell_factor: int")

        self.supercell_factor = supercell_factor
        self.size = size

        self.cell_filters = []
        self.atoms_filters = []

        self.add_cell_filter(SkewnessFilter(skew_threshold))
        self.all_cells = []
        self.add_cell_filter(EquivalentCellsFilter(self.all_cells))

        if self.size is not None:
            check_valid_conversion_matrix(self.size)
        self.skew_threshold = skew_threshold
        self.templates = None
        self.db_name = db_name
        self.db = connect(db_name)
        self.unit_cell = list(self.db.select(name='unit_cell'))[0].toatoms()
        self._set_based_on_setting()
        self._append_templates_from_db()

    def __str__(self):
        """Print a summary of the class."""
        msg = "=== TemplateAtoms ===\n"
        msg += "Supercell factor: {}\n".format(self.supercell_factor)
        msg += "Skewed threshold: {}\n".format(self.skew_threshold)
        msg += "Template sizes:\n"
        for size in self.templates['size']:
            msg += "{}\n".format(size)
        return msg

    @property
    def num_templates(self):
        return len(self.templates['atoms'])

    def add_cell_filter(self, cell_filter):
        """Attach a new Cell filter."""
        if not isinstance(cell_filter, CellFilter):
            raise TypeError("filter has to be an instance of CellFilter")
        self.cell_filters.append(cell_filter)

    def add_atoms_filter(self, at_filter):
        """Attach a new Atoms filter."""
        if not isinstance(at_filter, AtomsFilter):
            raise TypeError("filter has to be an instance of AtomsFilter")
        self.atoms_filters.append(at_filter)

    def clear_filters(self):
        """Remove all filters."""
        self.cell_filters = []
        self.atoms_filters = []

    def is_valid(self, atoms=None, cell=None):
        """
        Check the validity of the template.

        Return `True` if templates are valid according to the attached filters.

        Parameters:

        atoms: Atoms object

        cell: unit cell vector
        """
        if atoms is None and cell is None:
            msg = "At least one of `atoms` or `cell` must be specified."
            raise ValueError(msg)

        cell_valid = True
        if cell is not None:
            cell_valid = all([filter(cell) for filter in self.cell_filters])

        if not cell_valid:
            return False

        atoms_valid = True
        if atoms is not None:
            atoms_valid = all([f(atoms) for f in self.atoms_filters])

        return cell_valid and atoms_valid

    def get_size(self):
        """Get size of the templates."""
        return self.templates['size']

    def get_atoms(self, uid, return_size=False):
        """Return atoms at position."""
        if return_size:
            assert is_3x3_matrix(self.templates['size'][uid])
            return self.templates['atoms'][uid], self.templates['size'][uid]

        return self.templates['atoms'][uid]

    def get_uid_with_given_size(self, size, generate_template=False):
        """Get the UID of the template with given size.

        Parameters:

        size: list of length 3

        generate_template: bool (optional)
            If *True*, generate a new template if a template with matching
            size is not found.
        """
        if not is_3x3_matrix(size):
            raise ValueError("Expect the size to be a list of 3x3 matrix")

        try:
            uid = self.templates['size'].index(size)
            return uid
        except IndexError:
            pass

        if not generate_template:
            raise ValueError("There is no template with size = {}."
                             "".format(size))

        # get dims based on the passed atoms and append.
        _logger("Template that matches the specified size not found. "
                "Generating...")
        check_valid_conversion_matrix(size)
        unit_cell = self.unit_cell
        self.templates['atoms'].append(unit_cell*size)
        self.templates['size'].append(size)
        self._check_templates_datastructure()

        return len(self.templates['atoms']) - 1

    def get_uid_matching_atoms(self, atoms, generate_template=False):
        """Get the UID for the template matching atoms.

        Parameters:

        atoms: Atoms object
            structure to compare its size against template atoms

        generate_template: bool (optional)
            If *True*, generate a new template if a template with matching
            size is not found.
        """
        shape = atoms.get_cell_lengths_and_angles()
        for uid, template in enumerate(self.templates['atoms']):
            shape_template = template.get_cell_lengths_and_angles()
            if np.allclose(shape, shape_template):
                return uid

        if not generate_template:
            raise RuntimeError("There is no template that matches the shape "
                               "of given atoms object")

        # get dims based on the passed atoms and append.
        _logger("Template that matches the size of passed atoms not found. "
                "Generating...")
        size = self._get_conversion_matrix(atoms)
        assert is_3x3_matrix(size)

        unit_cell = self.unit_cell

        self.templates['atoms'].append(make_supercell(unit_cell, size))
        self.templates['size'].append(list(size))
        self._check_templates_datastructure()
        return len(self.templates['atoms']) - 1

    def _set_based_on_setting(self):
        """Construct templates based on arguments specified."""
        if self.size is None:
            self.supercell_factor = int(self.supercell_factor)
            self.templates = self._generate_template_atoms()
            if not self.templates['atoms']:
                raise RuntimeError("No template atoms with matching criteria")
        else:
            assert is_3x3_matrix(self.size)
            # if size and supercell_factor are both specified,
            # size will be used
            unit_cell = self.unit_cell
            self.templates = {'atoms': [make_supercell(unit_cell, self.size)],
                              'size': [self.size]}
        self._check_templates_datastructure()

    def _append_templates_from_db(self):
        if not os.path.isfile(self.db_name):
            return
        for row in self.db.select(name='template'):
            found = False
            for i, _ in enumerate(self.templates['atoms']):
                size = str2nested_list(row.size)
                assert is_3x3_matrix(size)
                if (self.templates['size'][i] == size):
                    found = True
                    break

            if not found:
                atoms = make_supercell(self.unit_cell, size)
                self.templates['atoms'].append(atoms)
                self.templates['size'].append(size)
        self._check_templates_datastructure()

    def _generate_template_atoms(self):
        """Generate all template atoms up to a certain multiplicity factor."""
        templates = {'atoms': [], 'size': []}
        # case 1: size of the cell is given
        if self.size is not None:
            assert is_3x3_matrix(self.size)

            atoms = make_supercell(self.unit_cell, self.size)
            templates['atoms'].append(atoms)
            templates['size'].append(self.size)
            return templates

        # case 2: supercell_factor is given
        for size in product(range(1, self.supercell_factor+1), repeat=3):
            # Skip cases where the product of factors is larger than the
            # supercell factor.
            if np.prod(size) > self.supercell_factor:
                continue
            matrix = np.diag(size)
            atoms = make_supercell(self.unit_cell, matrix)

            if self.is_valid(atoms=atoms, cell=atoms.get_cell()):
                templates['atoms'].append(atoms)
                templates['size'].append(matrix.tolist())
                self.all_cells.append(atoms.get_cell())
        return templates

    def _construct_templates_from_supercell(self, size):
        """Construct subcells based on a supercell."""
        templates = {'atoms': [], 'size': []}

        # Select the first unit cell entry in the DB
        ucell = self.unit_cell
        V = ucell.get_volume()

        supercell = ucell*size
        com = supercell.get_center_of_mass()

        dists = supercell.get_positions() - com
        lengths_squared = np.sum(dists**2, axis=1)
        closest = np.argmin(lengths_squared)
        pos = supercell.get_positions()
        origo = pos[closest, :]
        pos -= origo

        # Construct all templates
        symb = supercell[closest].symbol
        indices = [atom.index for atom in supercell if atom.symbol == symb]
        indices.remove(closest)

        distances = supercell.get_distances(closest, indices)
        sorted_indices = np.argsort(distances).tolist()
        indices = [indices[i] for i in sorted_indices]

        inv_cell = np.linalg.inv(ucell.get_cell().T)

        scaled_pos = inv_cell.dot(pos.T).T
        filtered_indices = []
        for indx in indices:
            vec = np.round(scaled_pos[indx, :])
            if np.allclose(vec.astype(np.int32), vec):
                filtered_indices.append(indx)
        indices = filtered_indices

        for i1, i2, i3 in combinations(indices, r=3):
            if i1 == i2 or i1 == i3 or i2 == i3:
                continue

            try:
                v1 = pos[i1, :]
                v2 = pos[i2, :]
                v3 = pos[i3, :]
                v1 = inv_cell.dot(v1)
                v2 = inv_cell.dot(v2)
                v3 = inv_cell.dot(v3)
                atoms = cut(ucell, a=v1, b=v2, c=v3)
            except np.linalg.linalg.LinAlgError:
                # Can't construct a 3D cell
                continue
            new_vol = atoms.get_volume()

            if new_vol > V*self.supercell_factor:
                continue

            if not self.is_valid(atoms=atoms, cell=atoms.get_cell()):
                continue

            size = [[int(np.round(x)) for x in v1.tolist()],
                    [int(np.round(x)) for x in v2.tolist()],
                    [int(np.round(x)) for x in v3.tolist()]]
            templates['atoms'].append(atoms)
            templates['size'].append(size)
        return templates

    def _get_conversion_matrix(self, atoms):
        """Return the conversion matrix factor."""
        unit_cell = self.unit_cell

        small_cell = unit_cell.get_cell()
        inv_cell = np.linalg.inv(small_cell)

        large_cell = atoms.get_cell()
        size_factor = large_cell.dot(inv_cell)
        scale_int = size_factor.round(decimals=0).astype(int)
        if np.allclose(size_factor, scale_int):
            check_valid_conversion_matrix(scale_int)
            return scale_int.tolist()

        raise ValueError("The passed atoms object cannot be described by "
                         "repeating of the unit cells. Scale factors found "
                         "{}".format(size_factor))

    def _internal_distances_are_equal(self, atoms1, atoms2):
        """Check if all internal distances are equivalent."""
        if len(atoms1) != len(atoms2):
            return False
        dist1 = []
        dist2 = []
        for ref in range(len(atoms1)-1):
            remaining = range(0, len(atoms1))
            dist1 += atoms1.get_distances(ref, remaining).tolist()
            dist2 += atoms2.get_distances(ref, remaining).tolist()
        return np.allclose(sorted(dist1), sorted(dist2))

    def random_template(self, max_supercell_factor=1000):
        """
        Select a random template atoms.

        Parameters:

        max_supercell_factor: int
            Maximum supercell_factor the returned object can have
        """
        found = False
        num = 0
        while not found:
            num = choice(range(len(self.templates['atoms'])))
            factor = np.prod(self.templates['size'][num])
            if factor <= max_supercell_factor:
                found = True
        return num

    def _get_max_min_diag_ratio(self, atoms):
        """Return the ratio between the maximum and the minimum diagonal."""
        diag_lengths = []
        cell = atoms.get_cell().T
        for w in product([-1, 0, 1], repeat=3):
            if np.allclose(w, 0):
                continue
            diag = cell.dot(w)
            length = np.sqrt(diag.dot(diag))
            diag_lengths.append(length)
        max_length = np.max(diag_lengths)
        min_length = np.min(diag_lengths)
        return max_length/min_length

    def weighted_random_template(self):
        """Select a random template atoms with a bias towards a cubic cell.

        The bias is towards cells that have similar values for x-, y- and
        z-dimension sizes.
        """
        p_select = []
        for atoms in self.templates['atoms']:
            ratio = self._get_max_min_diag_ratio(atoms)
            p = np.exp(-4.0*(ratio-1.0)/self.skew_threshold)
            p_select.append(p)
        p_select = np.array(p_select)
        p_select /= np.sum(p_select)

        cum_prob = np.cumsum(p_select)
        rand_num = np.random.rand()
        indx = np.argmax(cum_prob > rand_num)
        return indx

    def _check_templates_datastructure(self):
        """Fails if the datastructure is inconsistent."""
        num_entries = len(self.templates['atoms'])
        for _, v in self.templates.items():
            assert len(v) == num_entries

    def _transformation_matrix_with_given_volume(self, diag_A, diag_B,
                                                 off_diag_range):
        """
        This function creates integer matrices with a given determinant.
        It utilise that determinant of the dot product between an upper
        triangular matrix and a lower triangular matrix is equal to the
        product of the diagonal elements in the two matrices.

        C = A.dot(B)

            [a11, a, b]        [b11, 0, 0]
        A = [0, a22, c],  B =  [d, b22, 0]
            [0, 0, a33]        [e, f, b33]

        the determinant of C is then equal to
        det(C) = a11*a22*a33*b11*b22*b33.

        Parameters:

        diag_A: list of length 3
            3 integers representing the diagonal in the A matrix
        diag_B: list of length 3
            3 integers representing the diagonal in the B matrix
        off_diag_range: int
            The off diagonal elements are randomly chosen integers in the
            range [-off_diag_range, off_diag_range]
        """

        A = np.diag(diag_A)  # Upper triangular matrix
        B = np.diag(diag_B)  # Lower triangular matrix

        # Generate random off diagonal elements
        off_diag = np.random.randint(-off_diag_range, off_diag_range,
                                     size=6)

        A[0, 1] = off_diag[0]
        A[0, 2] = off_diag[1]
        A[1, 2] = off_diag[2]
        B[1, 0] = off_diag[3]
        B[2, 0] = off_diag[4]
        B[2, 1] = off_diag[5]

        C = A.dot(B)
        return C

    def get_template_given_volume(self, diag_A=[1, 1, 1], diag_B=[1, 1, 1],
                                  off_diag_range=2):
        """
        Generate a single template with a given volume. See doc string
        of `clease.template_atoms._transformation_matrix_with_given_volume`
        for details of the algorithm. The selected determinant is given by
        np.prod(diag_A)*np.prod(diag_B).

        Example:
        If a template with a volume equal to 2 times the unit cell, the
        following combinations of diag_A and diag_B would to the job
        diag_A = [2, 1, 1] and diag_B = [1, 1, 1]
        diag_A = [1, 2, 1] and diag_B = [1, 1, 1]
        diag_A = [1, 1, 2] and diag_B = [1, 1, 1]
        and all cases where diag_A and diag_B in the above list is swapped.

        If a template with a volume equal to 4 times the volume of the unit
        cell one can either have diag_A = [2, 2, 1] and diag_A = [4, 1, 1]
        works (diag_B = [1, 1, 1]).

        Parameters:
        diag_A: list int
            List of 3 integers representing the diagonal of the upper
            triangular matrix
        diag_B: list int
            List of 3 integers representing the diagonal of the lower
            triangular matrix
        off_diag_range: int
            The off diagonal elements are randomly chosen in the range
            [-off_diag_range, off_diag_range]
        """
        matrix = self._transformation_matrix_with_given_volume(
            diag_A, diag_B, off_diag_range)
        return make_supercell(self.unit_cell, matrix)

    def get_templates_given_volume(self, diag_A=[1, 1, 1], diag_B=[1, 1, 1],
                                   off_diag_range=2, num_templates=10):
        """
        Generate a given number of random templates with a fixed volume
        See also `clease.template_atoms.get_template_given_volume`.

        Parameters:

        diag_A: list of int
            List of 3 integers representing the diagonal of the upper
            triangular matrix

        diag_B: list of int
            List of 3 integers repreenting the diagonal in the lower
            triangular matrix

        off_diag_range: int
            The off diagonal elements are randomly chosen in the range
            [-off_diag_range, off_diag_range]

        num_templates: int
            Number of templates to generate
        """
        max_attempts = 1000
        matrices = []
        int_matrices = []

        # Use the equivalent cell filter to trace already selected templates
        equiv_filter = EquivalentCellsFilter(matrices)

        counter = 0
        ucell = self.unit_cell.get_cell()
        while len(int_matrices) < num_templates and counter < max_attempts:
            counter += 1
            matrix = self._transformation_matrix_with_given_volume(
                diag_A, diag_B, off_diag_range)

            sc = matrix.dot(ucell)

            # Check if this matrix can be obtained with a unitary
            # transformation of any of the other
            valid = self.is_valid(cell=sc)

            if valid and equiv_filter(sc):
                int_matrices.append(matrix)
                matrices.append(sc)

        templates = []
        for mat in int_matrices:
            templates.append(make_supercell(self.unit_cell, mat))
        return templates


def is_3x3_matrix(array):
    return np.array(array).shape == (3, 3)


def check_valid_conversion_matrix(array):
    """
    Make sure that we have a right-handed coordinate system.
    Raise a ValueError if the matrix is not valid.
    """
    determinant = np.linalg.det(array)
    if determinant < 0.0:
        raise ValueError("The determinant of the size matrix is less than "
                         "zero (got {}). For a right "
                         "coordinate system, we need a positive "
                         "determinant!".format(determinant))
