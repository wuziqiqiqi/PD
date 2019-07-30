# distutils: language = c++

from clease.cython.ce_updater cimport CEUpdater

cdef class PyCEUpdater:
    """
    Cython wrapper for the C++ class
    """
    cdef CEUpdater *thisptr
    cdef object bc
    cdef object corr_func
    cdef object eci

    def __cinit__(self):
        self.thisptr = new CEUpdater()

    def __dealloc__(self):
        del self.thisptr

    def __init__(self, atoms, bc, corr_func, eci):
        self.bc = bc
        self.corr_func = corr_func
        self.eci = eci
        self.thisptr.init(atoms, bc, corr_func, eci)

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

    def set_ecis(self, ecis):
        self.thisptr.set_ecis(ecis)

    def get_singlets(self):
        return self.thisptr.get_singlets()

    def get_energy(self):
        return self.thisptr.get_energy()

    def get_symbols(self):
        return self.thisptr.get_symbols()

    def set_num_threads(self, num_threads):
        self.thisptr.set_num_threads(num_threads)
