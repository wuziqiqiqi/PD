import numpy as np
from clease.jit import int32, jitclass

spec = [
    ('global_index', int32[:]),
    ('num_rows', int32),
    ('num_cols', int32),
    ('matrix_elements', int32[:])
]


@jitclass(spec)
class ColumnSparseMatrix(object):
    def __init__(self, num_rows, columns):
        self.global_index = -np.ones(np.max(columns)+1, dtype=np.int32)

        # Remap the column index
        for i in range(len(columns)):
            self.global_index[columns[i]] = i

        self.num_rows = num_rows
        self.num_cols = len(columns)
        self.matrix_elements = np.zeros(self.num_rows*self.num_cols,
                                        dtype=np.int32)

    def _index_in_element_array(self, row, col):
        return row*self.num_cols + self.global_index[col]

    def insert(self, row, col, value):
        if self.global_index[col] == -1:
            raise ValueError("Invalid column!")
        elem_index = self._index_in_element_array(row, col)
        self.matrix_elements[elem_index] = value

    def get(self, row, col):
        return self.matrix_elements[self._index_in_element_array(row, col)]
