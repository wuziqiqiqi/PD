# distutils: language = c++
# cython: c_string_type=str, c_string_encoding=ascii
import numpy as np
cimport numpy as np  # Initialize the Numpy API
from libcpp cimport bool

np.import_array()
include "pyce_updater.pyx"
include "py_cluster.pyx"
include "py_atoms.pyx"

cdef extern from "additional_tools.hpp":
    cpdef bool has_parallel()

# Files that use the Numpy Array API are included here
cdef extern from "with_numpy/ce_updater.cpp":
    pass
