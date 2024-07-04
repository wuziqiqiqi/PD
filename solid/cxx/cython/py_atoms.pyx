"""Expose the C++ Atoms object to Python"""
# distutils: language = c++
#import copy
from typing import List, Sequence
from atoms cimport Atoms
from libcpp.vector cimport vector
# 4-vector, as per the C++ definiton
from four_vector cimport FourVector as cppFourVector
# The CLEASE Python definiton of the 4-vector
from clease.datastructures.four_vector import FourVector
from clease.datastructures.system_changes import SystemChange
import ase

cdef class CppAtoms:
    cdef Atoms *thisptr

    def __cinit__(self, atoms: ase.Atoms, four_vectors: Sequence[FourVector]):
        # Extract the atomic numbers directly from the Atoms object,
        # and make a list-copy.
        #numbers = list(atoms.get_atomic_numbers())
        self.thisptr = new Atoms(atoms, four_vectors)

    def __dealloc__(self):
        del self.thisptr

    def get_numbers(self) -> List[int]:
        return list(self.thisptr.get_numbers())

    def get_symbols(self) -> List[str]:
        return list(self.thisptr.get_symbols())

    def get_four_vectors(self) -> List[FourVector]:
        cdef vector[cppFourVector] vectors
        vectors = self.thisptr.get_four_vectors()

        L = vectors.size()
        return [_to_clease_fourvec(vectors[i]) for i in range(L)]

    @property
    def Ns(self) -> int:
        return self.thisptr.Ns

    @property
    def Nx(self) -> int:
        return self.thisptr.Nx

    @property
    def Ny(self) -> int:
        return self.thisptr.Ny

    @property
    def Nz(self) -> int:
        return self.thisptr.Nz

    def get_1d_index(self, vec: FourVector) -> int:
        cdef cppFourVector cpp_vec
        cdef int idx
        cpp_vec = _to_cpp_fourvec(vec)
        idx = self.thisptr.get_1d_index(cpp_vec)
        return idx

    def apply_change(self, change: SystemChange) -> None:
        self.thisptr.apply_change(change)

    def undo_change(self, change: SystemChange) -> None:
        self.thisptr.undo_change(change)

cdef _to_clease_fourvec(cppFourVector vec):
    """Convert a c++ FourVector into a CLEASE Python FourVector object."""
    return FourVector(vec.ix, vec.iy, vec.iz, vec.sublattice)

cdef cppFourVector _to_cpp_fourvec(vec: FourVector):
    cdef cppFourVector fourvec
    fourvec = cppFourVector()
    fourvec.ix = vec.ix
    fourvec.iy = vec.iy
    fourvec.iz = vec.iz
    fourvec.sublattice = vec.sublattice
    return fourvec
