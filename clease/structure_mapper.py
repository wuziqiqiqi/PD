from ase.atoms import Atoms, Cell
import spglib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Tuple


class TransformInfo(object):
    """
    Class for holding information about snap transformation
    """
    def __init__(self):
        self.displacements = []
        self.strain = []

    def todict(self) -> dict:
        return {
            'displacements': list(self.displacements),
            'strain': list(self.strain)
        }


class StructureMapper(object):
    """
    Class that refines a relaxed structure to a corresponding "ideal"
    lattice.

    :param symprec: Symmetry precision (see refine_cell in spglib)
    :param angle_tol: Angle tolerance (see refine_cell in spglib)
    """
    def __init__(self, symprec: float = 0.1, angle_tol: float = 5.0):
        self.symprec = symprec
        self.angle_tol = angle_tol

    def refine(self, atoms: Atoms) -> Atoms:
        """
        Refine a relaxed atoms object. This method relies on symmetry finding.
        It handles distortions to some extent, but for heavily distorted
        systems, it will most likely fail.

        :param atoms: Relaxed atoms object
        """
        cell = np.array(atoms.get_cell())
        scaled_pos = atoms.get_scaled_positions()
        numbers = np.array(atoms.numbers)

        result = spglib.refine_cell(
            (cell, scaled_pos, numbers),
            symprec=self.symprec, angle_tolerance=self.angle_tol)

        if result is None:
            raise RuntimeError("Could not refine cell")

        lattice, scaled_pos, numbers = result
        return Atoms(numbers=numbers, cell=lattice,
                     scaled_positions=scaled_pos, pbc=[1, 1, 1])

    def _extend_atoms(self, atoms: Atoms, buffer: float = 0.01) -> Atoms:
        """
        Adds periodic images within buffer

        :param atoms: Atoms object that will be extended
        :param buffer: Buffer in scaled coordinates
        """
        atoms_cpy = atoms.copy()
        for atom in atoms_cpy:
            atom.tag = atom.index
        atoms_cpy = atoms_cpy*(3, 3, 3)
        scaled_pos = atoms_cpy.get_scaled_positions()
        filtered_pos = []
        filtered_numbers = []
        filtered_tags = []
        for i in range(scaled_pos.shape[0]):
            x = scaled_pos[i, :]
            if np.all(x > 1.0/3.0 - buffer) and np.all(x < 2.0/3.0 + buffer):
                filtered_pos.append(x)
                filtered_numbers.append(atoms_cpy.numbers[i])
                filtered_tags.append(atoms_cpy[i].tag)

        filtered_pos = np.array(filtered_pos)*3.0 - 1.0
        cartesian_pos = atoms.get_cell().dot(filtered_pos.T).T
        return Atoms(numbers=filtered_numbers, positions=cartesian_pos,
                     tags=filtered_tags)

    def strain(self, cell1: Cell, cell2: Cell) -> np.ndarray:
        """
        Calculate the strain tensor to map cell1 into cell2.
        In the following the 3x3 matrix where each cell vector is a
        column, is denoted C_1 and C_2 for cell1 and cell2, respectively.
        There is a transformation matrix P such that C_2 = PC_1. Thus,
        P = C_2C_1^{-1}. The strain tensor is defined as E = 0.5*(P^TP - I),
        where I is the identity matrix.

        :param cell1: First cell
        :param cell2: Second cell
        """
        P = cell2.dot(np.linalg.inv(cell1))
        return 0.5*(P.T.dot(P) - np.eye(3))

    def snap_to_lattice(self, atoms: Atoms, template: Atoms
                        ) -> Tuple[Atoms, TransformInfo]:
        """
        Snaps atoms to an ideal lattice. If the number of atoms in the
        relaxed atoms object is less than the number of atoms in the template,
        the method will add vacancies on sites that are not matched.

        :param atoms: Relaxed atoms object
        :param template: Ideal lattice
        """
        strain = self.strain(atoms.get_cell(), template.get_cell())
        atoms.set_cell(template.get_cell(), scale_atoms=True)
        extended_template = self._extend_atoms(template)
        pos1 = atoms.get_positions()
        pos2 = extended_template.get_positions()

        # Snap atoms to ideal sites such that collective distance
        # is minimized
        distances = cdist(pos1, pos2)
        row, assignment = linear_sum_assignment(distances)
        displacements = distances[row, assignment]
        template_copy = extended_template.copy()
        template_copy.numbers = np.zeros(len(template_copy))
        for i in range(len(atoms)):
            template_copy[assignment[i]].symbol = atoms[i].symbol

        snapped = template.copy()
        populated = np.zeros(len(snapped), dtype=np.uint8)
        for i, atom in enumerate(template_copy):
            if not populated[atom.tag]:
                snapped[atom.tag].symbol = atom.symbol
                populated[i] = 1
            else:
                if snapped[atom.tag].symbol == 'X':
                    snapped[atom.tag].symbol = atom.symbol

        transform_info = TransformInfo()
        transform_info.displacements = displacements
        transform_info.strain = strain
        return snapped, transform_info
