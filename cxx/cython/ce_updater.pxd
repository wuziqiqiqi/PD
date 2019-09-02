# distutils: language = c++

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map as map_cpp

cdef extern from "ce_updater.hpp":
  cdef cppclass CEUpdater:
      CEUpdater() except +

      # Initialize the object
      void init(object atoms, object BC, object corrFunc, object eci, object cluster_info) except +

      # Clear update history
      void clear_history()

      # Undo all chages since last call to clear_history
      void undo_changes()

      # Update the correlation functions
      void update_cf(object system_changes) except +

      double get_energy()

      double calculate(object system_changes) except +

      object get_cf()

      void set_eci(object eci)

      object get_singlets()

      const vector[string]& get_symbols() const

      void set_num_threads(unsigned int num_threads)

      void get_changes(vector[string] &symbs, vector[unsigned int] &changed_sites) except +

      void calculate_cf_from_scratch(const vector[string] &cf_names, map_cpp[string, double] &cf)

      void set_atoms(object atoms) except+
