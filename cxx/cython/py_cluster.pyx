"""Expose the C++ Cluster object to Python"""
# distutils: language = c++


from cluster cimport Cluster

cdef class CppCluster:
    cdef Cluster *thisptr

    def __init__(self, cluster_obj):
        """Construct the C++ Cluster from a regular CLEASE
        python cluster object.
        
        Parameters:
        
        :cluster_obj: Instance of a clease.cluster.Cluster object
        """
        self.thisptr = new Cluster(cluster_obj)

    def __dealloc__(self):
        del self.thisptr

    def get_all_decoration_numbers(self, n_basis_funcs):
        return self.thisptr.get_all_decoration_numbers(n_basis_funcs)

    def get_size(self):
        return self.thisptr.get_size()
