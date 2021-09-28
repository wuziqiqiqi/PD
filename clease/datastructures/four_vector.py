from collections import Counter
from itertools import product
from typing import NamedTuple, Tuple, List
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ase.atoms import Cell
from ase import Atoms
import attr

__all__ = ('FourVector', 'construct_four_vectors')


@attr.s(frozen=True, eq=True)
class FourVector:
    ix: int = attr.ib(validator=attr.validators.instance_of(int))
    iy: int = attr.ib(validator=attr.validators.instance_of(int))
    iz: int = attr.ib(validator=attr.validators.instance_of(int))
    sublattice: int = attr.ib(validator=attr.validators.instance_of(int))

    def to_cartesian(self, prim: Atoms) -> np.ndarray:
        """
        Convert the four vector into cartesian coordinates

        :param prim: Primitive cell that defines the lattice
        """
        cell = prim.get_cell()
        return cell.T.dot([self.ix, self.iy, self.iz]) + prim[self.sublattice].position

    def to_scaled(self, prim: Atoms) -> np.ndarray:
        """
        Convert the four vector into scaled coordinates

        :param prim: Primitive cell that defines the lattice
        """
        return np.array([self.ix, self.iy, self.iz]) + \
            prim.get_scaled_positions()[self.sublattice, :]

    def to_tuple(self) -> Tuple[int]:
        """Get the tuple representation of the four-vector
        """
        return attr.astuple(self)


class _Box(NamedTuple):
    """
    Class defining a box. The box is defined by storing the position vector of
    the start (lower) and end (upper) of the diagonal. The names lower and upper
    refers to "lower corner" and "upper corner", respectively.
    """
    lower: np.ndarray
    upper: np.ndarray


def _supercell_is_integer_multiple(prim: Cell, supercell: Cell, tol: float = 1e-8) -> bool:
    """
    Return True ff C_{super} = C_{prim}P and P is an integer matrix. The columns of C_{super}
    is the cell vectors of the supercell and the columns of C_{prim} are the cell vectors of
    the primitive cell

    :param prim: Primitive cell
    :param supercell: Supercell
    :param tol: Tolerance for comparison
    """
    P = np.linalg.solve(prim.array.T, supercell.array.T)
    P_int = np.round(P)
    return np.max(np.abs(P - P_int)) < tol


def _make_grid(bbox: _Box, prim: Atoms) -> Tuple[List[FourVector], np.ndarray]:
    """
    Make a grid bounded by bbox using repetitions of the atomic positions in prim

    :param bbox: Bounding box for the grid
    :param prim: Primitive cell that defines the sublattice

    :return:
        four_vecs: List of FourVector of length N (where N is the number of grid
                   points)
        scaled_pos: Nx3 numpy array. Each row of scaled_pos corresponds to the
            four-vector in four_vecs
    """
    four_vecs = []
    scaled_pos = []

    ranges = [range(bbox.lower[i], bbox.upper[i] + 1) for i in range(3)] + [range(len(prim))]
    for ix, iy, iz, l in product(*ranges):
        fv = FourVector(ix, iy, iz, l)
        four_vecs.append(fv)
        scaled_pos.append(fv.to_scaled(prim))

    # Convert to 2D numpy array and perform a sanity check on the result
    scaled_pos = np.array(scaled_pos)
    assert scaled_pos.shape == (len(four_vecs), 3)
    return four_vecs, scaled_pos


def construct_four_vectors(prim: Atoms, supercell: Atoms) -> List[FourVector]:
    """
    Calculate the all four vector corresponding to the passed Atoms object. A four-vector u is
    defined by such that the position of an atom is given by C.dot(u[:3]) + prim[u[3]].position
    where the cell vectors of the primitive cell are column in the C matrix.

    The cell of the supercell must be related to the cell in prim my an integer matrix
    transformation. C_{super} = C_{prim}P. See for example `make_supercell`in ASE.

    :param prim: Primitive cell
    :param supercell: Supercell
    """
    if not _supercell_is_integer_multiple(prim.get_cell(), supercell.get_cell()):
        raise ValueError("The supercell must be an integer multiple of the primitive cell")

    if np.any(prim.get_scaled_positions() < 0.0):
        raise ValueError("Primitive cell must be wrapped")

    supercell_cpy = supercell.copy()
    supercell_cpy.set_cell(prim.get_cell())
    sc_scaled_pos = supercell_cpy.get_scaled_positions(wrap=False)

    bounding_box = _Box(lower=np.floor(np.min(sc_scaled_pos, axis=0)).astype(int),
                        upper=np.ceil(np.max(sc_scaled_pos, axis=0)).astype(int))

    # Fill the bounding box with a grid
    four_vecs_grid, scaled_pos_grid = _make_grid(bounding_box, prim)

    # Calculates pair-wise distances between the super-cell and the grid
    distances = cdist(sc_scaled_pos, scaled_pos_grid)

    # Assign exactly one atom to each grid point such that the total distance mismatch is minimize
    # (e.g. solving least linear sym assignment problem)
    _, cols = linear_sum_assignment(distances)

    # Calculate mean displacement
    four_vectors = [four_vecs_grid[c] for c in cols]

    # Sanity checks
    sub_lat_count = Counter(fv.sublattice for fv in four_vectors)

    if any(v != int(len(supercell_cpy) / len(prim)) for v in sub_lat_count.values()):
        raise RuntimeError("The four vectors are invalid as the number of atoms in each "
                           "sublattice does not match the number of primitive cells")

    if len(sub_lat_count) != len(prim):
        raise RuntimeError("The four vectors are invalid as the number of sublattice "
                           "detected does not match the number of sublattices in the "
                           "primitive cell")

    return four_vectors
