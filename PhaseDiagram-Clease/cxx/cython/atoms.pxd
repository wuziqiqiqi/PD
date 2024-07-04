"""Import the C++ definition of an Atoms object to Cython"""
# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string

from four_vector cimport FourVector

cdef extern from "atoms.hpp":
    cdef cppclass Atoms:
        Atoms(object atoms, object four_vectors) except +

        int Ns, Nx, Ny, Nz

        vector[int] get_numbers() const
        vector[string] get_symbols() const

        vector[FourVector] get_four_vectors() const

        int get_1d_index(FourVector vec) const

        void apply_change(object single_change)
        void undo_change(object single_change)
