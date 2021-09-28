"""Import the C++ definition of a FourVector to Cython"""
# distutils: language = c++

cdef extern from "four_vector.hpp":
    cdef cppclass FourVector:
        FourVector()

        int ix, iy, iz, sublattice
