from clease.column_sparse_matrix import ColumnSparseMatrix
import numpy as np
from ase.test import must_raise


def test_insert_access():
    columns = np.array([0, 4, 7, 3])
    values = [
        (0, 4),
        (4, -1),
        (7, 4),
        (7, 6),
        (3, -1)
    ]

    num_rows = len(values)
    sp_mat = ColumnSparseMatrix(num_rows, columns)

    for row, val in enumerate(values):
        sp_mat.insert(row, val[0], val[1])
    for row, val in enumerate(values):
        assert sp_mat.get(row, val[0]) == val[1]

    with must_raise(ValueError):
        sp_mat.insert(2, 2, 1)


test_insert_access()
