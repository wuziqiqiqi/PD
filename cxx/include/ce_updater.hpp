#ifndef CE_UPDATER_H
#define CE_UPDATER_H
#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include "symbol_change.hpp"
#include "matrix.hpp"
#include "row_sparse_struct_matrix.hpp"
#include "cf_history_tracker.hpp"
#include <array>
#include <Python.h>
#include "cluster.hpp"
#include "named_array.hpp"
#include "symbols_with_numbers.hpp"
#include "basis_function.hpp"
#include "cluster_list.hpp"

// Read values from name_list
// name_list[symm_group][cluster_size] = vector of string variables of all the cluster names
typedef std::vector<std::vector<std::vector<std::string>>> name_list;

// Read values form cluster_list
// cluster_list[symm_group][cluster_size][indx] = vector of indices belonging to the cluster #indx.
typedef std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> cluster_list;
typedef std::vector<std::map<std::string, double>> bf_list;
typedef std::array<SymbolChange, 2> swap_move;
typedef std::unordered_map<std::string, Cluster> cluster_dict;

// typedef std::unordered_map<std::string,double> cf;
typedef NamedArray cf;

enum class Status_t
{
  READY,
  INIT_FAILED,
  NOT_INITIALIZED
};

typedef std::map<std::string, std::vector<int>> tracker_t;

/** Help structure used when the correlation functions have different decoration number */
struct ClusterMember
{
  int ref_indx;
  int sub_cluster_indx;
};

class CEUpdater
{
public:
  CEUpdater();
  ~CEUpdater();

  /** New copy. NOTE: the pointer has to be deleted */
  CEUpdater *copy() const;

  /** Initialize the object (cluster_info should contain duplication factors) */
  void init(PyObject *py_atoms, PyObject *settings, PyObject *corrFunc, PyObject *eci, PyObject *cluster_list);

  /** Change values of ECIs */
  void set_eci(PyObject *eci);

  /** Returns True if the initialization process was successfull */
  bool ok() const { return status == Status_t::READY; };

  /** Computes the energy based on the ECIs and the correlations functions */
  double get_energy();

  /** Returns the value of the singlets */
  void get_singlets(PyObject *npy_array) const;
  PyObject *get_singlets() const;

  /** Extracts basis functions from the cluster name */
  void get_basis_functions(const std::string &cluster_name, std::vector<int> &bfs) const;

  /** Updates the CF */
  void update_cf(PyObject *single_change);
  void update_cf(SymbolChange &single_change);

  /** Computes the spin product for one element */
  double spin_product_one_atom(int ref_indx, const Cluster &indx_list, const std::vector<int> &dec, int ref_id);
  // Calculate the change in spin product going from old_symb_id to new_symb_id
  double spin_product_one_atom_delta(int ref_indx, const Cluster &indx_list, const std::vector<int> &dec, int old_symb_id, int new_symb_id);

  /**
  Calculates the new energy given a set of system changes
  the system changes is assumed to be a python-list of tuples of the form
  [(indx1,old_symb1,new_symb1),(indx2,old_symb2,new_symb2)...]
  */
  double calculate(PyObject *system_changes);
  double calculate(swap_move &system_changes);
  double calculate(std::vector<swap_move> &sequence);
  double calculate(std::vector<SymbolChange> &sequence);

  /** Undo given number of steps */
  void undo_changes(int num_steps);

  /** Resets all changes */
  void undo_changes();

  /** Clears the history */
  void clear_history();

  /** Populates the given vector with all the cluster names */
  void flattened_cf_names(std::vector<std::string> &flattened);

  /** Returns the correlaation functions as a dictionary. Only the ones that corresponds to one of the ECIs */
  PyObject *get_cf();

  /** Returns the CF history tracker */
  const CFHistoryTracker &get_history() const { return *history; };

  /** Read-only reference to the symbols */
  const std::vector<std::string> &get_symbols() const { return symbols_with_id->get_symbols(); };

  /** Returns the cluster figures */
  const ClusterList &get_clusters() const { return clusters; };

  /** Return the cluster with the given name
   * The key in the map is the symmetry group
   */
  void get_clusters(const std::string &cname, std::map<unsigned int, const Cluster *> &clusters) const;
  void get_clusters(const char *cname, std::map<unsigned int, const Cluster *> &clusters) const;

  /** Returns the translation matrix */
  const RowSparseStructMatrix &get_trans_matrix() const { return trans_matrix; };

  /** Get the translation symmetry group of a site */
  unsigned int get_trans_symm_group(unsigned int indx) const { return trans_symm_group[indx]; };

  /** Sets the symbols */
  void set_symbols(const std::vector<std::string> &new_symbs);

  /** CE updater should keep track of where the atoms are */
  void set_atom_position_tracker(tracker_t *new_tracker) { tracker = new_tracker; };

  /** Set the number of threads to use during CF updating */
  void set_num_threads(unsigned int num) { cf_update_num_threads = num; };

  /** Find which sites have changed */
  void get_changes(const std::vector<std::string> &new_symbols, std::vector<unsigned int> &changed_sites) const;

  /** Calculate correlation functions from scratch */
  void calculate_cf_from_scratch(const std::vector<std::string> &cf_names, std::map<std::string, double> &cf);

  /** Set a new atoms object */
  void set_atoms(PyObject *atoms);

  static void py_changes2_symb_changes(PyObject *all_changes, std::vector<SymbolChange> &symb_changes);

private:
  void get_unique_indx_in_clusters(std::set<int> &unique_indx);

  /** Returns the maximum index occuring in the cluster indices */
  unsigned int get_max_indx_of_zero_site() const;

  unsigned int cf_update_num_threads{1};

  // std::vector<std::string> symbols;
  Symbols *symbols_with_id{nullptr};
  // std::vector<cluster_dict> clusters;
  ClusterList clusters;
  std::vector<int> trans_symm_group;
  std::vector<int> trans_symm_group_count;
  std::map<std::string, int> normalisation_factor;

  // bf_list basis_functions;
  BasisFunction basis_functions;

  Status_t status{Status_t::NOT_INITIALIZED};
  // Matrix<int> trans_matrix;
  RowSparseStructMatrix trans_matrix;
  std::map<std::string, int> ctype_lookup;
  NamedArray eci;
  std::map<std::string, std::string> cname_with_dec;
  std::vector<bool> is_background_index;
  bool ignore_background_indices{true};
  CFHistoryTracker *history{nullptr};
  PyObject *atoms{nullptr};
  tracker_t *tracker{nullptr}; // Do not own this pointer
  std::vector<std::string> singlets;

  // Cached number of non background sites
  void count_non_bkg_sites();
  int num_non_bkg_sites;

  /** Undos the latest changes keeping the tracker CE tracker updated */
  void undo_changes_tracker(int num_steps);

  /** Returns true if all decoration numbers are equal */
  bool all_decoration_nums_equal(const std::vector<int> &dec_num) const;

  /** Initialize the cluster name with decoration lookup */
  void create_cname_with_dec(PyObject *corrfuncs);

  /** Build a list over which translation symmetry group a site belongs to */
  void build_trans_symm_group(PyObject *single_term_clusters);

  /** Verifies that each ECI has a correlation function otherwise it throws an exception */
  bool all_eci_corresponds_to_cf();

  /** Verifies that each cluster name exists only in one symmetry group*/
  void verify_clusters_only_exits_in_one_symm_group();

  /** Reads the translation matrix */
  void read_trans_matrix(PyObject *py_trans_mat);

  /** Read background indices */
  void read_background_indices(PyObject *bkg_indices);

  /** Check if a move is a swap move */
  bool is_swap_move(const swap_move &move) const;

  /** Sort indices according to order */
  // static void sort_indices(std::vector<int> &indx, const std::vector<int> &order);
  static void sort_indices(int indices[], const std::vector<int> &order, unsigned int n_indices);
};
#endif
