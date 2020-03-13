# distutils: language = c++

from ce_updater cimport CEUpdater
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map as map_cpp
from cython.operator import dereference, postincrement

cdef class PyCEUpdater:
    """Cython wrapper for the C++ class"""
    cdef CEUpdater *thisptr
    cdef object settings
    cdef object corr_func
    cdef object eci

    def __cinit__(self):
        self.thisptr = new CEUpdater()

    def __dealloc__(self):
        del self.thisptr

    def __init__(self, atoms, settings, corr_func, eci, cluster_info):
        self.settings = settings
        self.corr_func = corr_func
        self.eci = eci
        self.thisptr.init(atoms, settings, corr_func, eci, cluster_info)

    def clear_history(self):
        self.thisptr.clear_history()

    def undo_changes(self):
        self.thisptr.undo_changes()

    def update_cf(self, system_changes):
        self.thisptr.update_cf(system_changes)

    def calculate(self, system_changes):
        return self.thisptr.calculate(system_changes)

    def get_cf(self):
        return self.thisptr.get_cf()

    def set_eci(self, eci):
        self.thisptr.set_eci(eci)

    def get_singlets(self):
        return self.thisptr.get_singlets()

    def get_energy(self):
        return self.thisptr.get_energy()

    def get_symbols(self):
        return self.thisptr.get_symbols()

    def set_num_threads(self, num_threads):
        self.thisptr.set_num_threads(num_threads)

    def get_changed_sites(self, atoms):
        symbs = [atom.symbol for atom in atoms]
        cdef vector[string] symb_vec

        for atom in atoms:
            symb_vec.push_back(atom.symbol)

        cdef vector[unsigned int] changed
        self.thisptr.get_changes(symb_vec, changed)
        return [changed[i] for i in range(changed.size())]

    def calculate_cf_from_scratch(self, atoms, cf_names):
        self.thisptr.set_atoms(atoms)

        cdef map_cpp[string, double] cf
        cdef vector[string] cname_vec

        # Transfer names to a C++ vector
        for name in cf_names:
            cname_vec.push_back(name)
        self.thisptr.calculate_cf_from_scratch(cname_vec, cf)

        # Transfer to python dict
        cf_dict = {}
        cdef map_cpp[string, double].iterator it = cf.begin()
        cdef map_cpp[string, double].iterator end = cf.end()

        # Transfer map[string, double] to dictionary
        while(it != end):
            cf_dict[dereference(it).first] = dereference(it).second
            postincrement(it)
        return cf_dict
