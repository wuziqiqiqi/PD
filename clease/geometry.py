"""Module for tools pertaining to geometry of atoms and cells."""
import numpy as np
from ase import Atoms

__all__ = ("max_sphere_dia_in_cell", "supercell_which_contains_sphere", "cell_wall_distances")


def supercell_which_contains_sphere(atoms: Atoms, diameter: float) -> Atoms:
    """Find the smallest supercell of an atoms object which can contain a sphere,
    using only repetitions of (nx, ny, nz) (i.e. a diagonal P matrix).

    The number of repetitions is stored in the info dictionary of the supercell under the
    "repeats" keyword, i.e. sc.info["repeats"] is the number of times the supercell was repeated.

    Args:
        atoms (Atoms): The atoms object to be enlarged.
        diameter (float): The diameter of the sphere which should be contained within the supercell.

    Returns:
        Atoms: The supercell which can contain a sphere of the given diameter.
    """
    cell = atoms.get_cell()
    dists = cell_wall_distances(cell)
    # Find the number of repetitions required for each cell
    # wall to be larger than the diameter, i.e. allowing
    # a sphere with the provided diameter to exist within the cell.
    reps = np.ceil(diameter / dists).astype(int)
    assert reps.shape == (3,)
    sc: Atoms = atoms * reps
    assert cell_wall_distances(sc.get_cell()).min() >= diameter

    sc.info["repeats"] = reps

    return sc


def cell_wall_distances(cell: np.ndarray) -> np.ndarray:
    """Get the distances from each cell wall to the opposite cell wall of the cell.
    Returns the distances in the order of the distances between the (b, c), (a, c) and (a, b)
    planes, such that the shorest vector corresponds to being limited by the a, b or c vector,
    respectively.

    Args:
        cell (np.ndarray): A (3 x 3) matrix which defines the cell parameters.
            Raises a ValueError if the cell shape is wrong.

    Returns:
        np.ndarray: The distance to each of the three cell walls.
    """
    if not cell.shape == (3, 3):
        raise ValueError(f"Cell should be (3, 3), got {cell.shape}")
    # Midpoint of the parallelogram spanned by the cell vectors
    midpoint = np.sum(cell / 2, axis=0)

    # Get the shortest distance between the midpoint and planes defined by the
    # ab, ac and bc vectors. Since the cell is a parallelogram, the 3 remaining planes will
    # have identical distances to the midpoint.
    # Assumption: One of the cell corners is at (0, 0, 0),
    # so 3 of the 6 planes will share the point (0, 0, 0).
    # PQ is the vector from our midpoint to the reference point in the plane.
    PQ = midpoint - [0, 0, 0]
    a, b, c = cell

    def _iter_distances() -> float:
        # Iterate all distances from cell wall to the midpoint
        # The ordering here ensures that the distances corresponds to the length
        # a, b, c in a cubic cell (in that order)
        # Alternatively: The vector which limits the size of the sphere
        # is determined by the argmin of this array.
        for v1, v2 in [(b, c), (a, c), (a, b)]:
            # Find the normal vector to the plane spanned by v1 and v2
            # Note: The astype(float) is to ensure that a cell defined by
            # integers still works.
            n = np.cross(v1, v2).astype(float)
            n /= np.linalg.norm(n)
            # Project onto the normal vector to get the shortest possible
            # distance.
            # Multiply by 2 to get the diameter.
            yield np.linalg.norm(PQ.dot(n)) * 2

    return np.array(list(_iter_distances()), dtype=float)


def max_sphere_dia_in_cell(cell: np.ndarray) -> float:
    """Find the diameter of the largest possible sphere which can be placed
    inside a cell.

    For example, how large of a Death Star could be built inside a given atoms object?

    Args:
        cell (np.ndarray): A (3 x 3) matrix which defines the cell parameters.
            Raises a ValueError if the cell shape is wrong.

    Returns:
        float: The diameter of the largest sphere which can fit within the cell.
    """
    # The smallest possible distance is the radius of the largest sphere we can have.
    return cell_wall_distances(cell).min()
