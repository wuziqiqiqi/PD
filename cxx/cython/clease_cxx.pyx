# distutils: language = c++
# cython: c_string_type=str, c_string_encoding=ascii
cimport numpy as np  # Initialize the Numpy API
np.import_array()
include "pyce_updater.pyx"

# Files that use the Numpy Array API are included here
cdef extern from "ce_updater.cpp":
    pass