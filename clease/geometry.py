"""Module for tools pertaining to geometry of atoms and cells."""
import itertools
import numpy as np

__all__ = ("max_sphere_dia_in_cell",)


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

    def _iter_distances() -> float:
        # Iterate all distances from cell wall to the midpoint
        for v1, v2 in itertools.combinations(cell, 2):
            # Find the normal vector to the plane spanned by v1 and v2
            # Note: The astype(float) is to ensure that a cell defined by
            # integers still works.
            n = np.cross(v1, v2).astype(float)
            n /= np.linalg.norm(n)
            # Project onto the normal vector to get the shortest possible
            # distance.
            # Multiply by 2 to get the diameter.
            yield np.linalg.norm(PQ.dot(n)) * 2

    # The smallest possible distance is the radius of the largest sphere we can have.
    return min(_iter_distances())
