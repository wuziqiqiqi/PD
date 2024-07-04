"""Import the C++ definition of a cluster to Cython"""
# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "cluster.hpp":
    cdef cppclass Cluster:
        Cluster(object info_dict) except +

        vector[vector[int]] get_all_decoration_numbers(int n_basis_funcs) const

        int get_size() const
