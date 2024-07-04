#ifndef CE_UPDATER_H
#define CE_UPDATER_H
#include <Python.h>

#include <array>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "basis_function.hpp"
#include "cf_history_tracker.hpp"
#include "cluster.hpp"
#include "cluster_list.hpp"
#include "cluster_name.hpp"
#include "matrix.hpp"
#include "named_array.hpp"
#include "row_sparse_struct_matrix.hpp"
#include "symbol_change.hpp"
#include "symbols_with_numbers.hpp"

typedef std::vector<std::map<std::string, double>> bf_list;
typedef std::array<SymbolChange, 2> swap_move;

typedef NamedArray cf;

enum class Status_t {
    READY,
    INIT_FAILED,
    NOT_INITIALIZED,
};

/* Collection of pointers to a cluster and equivalent decoration vector.
nullptr is used to indicate the Cluster is not used, generally for a given
translational symmetry group.*/
struct ClusterCache {
    Cluster *cluster_ptr{nullptr};
    equiv_deco_t *equiv_deco_ptr{nullptr};
    double normalization{1.0};
};

/** Help structure used when the correlation functions have different decoration number */
struct ClusterMember {
    int ref_indx;
    int sub_cluster_indx;
};
/* Cache structure which contains objects that are useful for the updating of the spin product */
struct SpinProductCache {
    int ref_indx;        // The current reference index
    int original_index;  // The original "untranslated" index.
    /* A cache of the relevant translation matrix row
    corresponding to the ref_indx */
    int *trans_matrix_row;
    unsigned int new_symb_id;
    unsigned int old_symb_id;
};

class CEUpdater {
   public:
    CEUpdater();
    ~CEUpdater();

    /** Initialize the object (cluster_info should contain duplication factors) */
    void init(PyObject *py_atoms, PyObject *settings, PyObject *corrFunc, PyObject *eci,
              PyObject *cluster_list);

    /** Change values of ECIs */
    void set_eci(PyObject *eci);

    /** Returns True if the initialization process was successfull */
    bool ok() const {
        return status == Status_t::READY;
    };

    /** Compute the energy based on the ECIs and the correlations functions */
    double get_energy();

    /** Returns the value of the singlets */
    void get_singlets(PyObject *npy_array) const;
    PyObject *get_singlets() const;

    /** Updates the CF */
    void update_cf(PyObject *single_change);
    void update_cf(SymbolChange &single_change);

    /* Locate the original index for the "untranslated" site, given a
    reference index, i.e. the inverse operation of the translation matrix.*/
    int get_original_index(int ref_indx) const;

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

    /** Returns the correlaation functions as a dictionary. Only the ones that corresponds to one of
     * the ECIs */
    PyObject *get_cf();

    /** Returns the CF history tracker */
    const CFHistoryTracker &get_history() const {
        return *history;
    };

    /** Read-only reference to the symbols */
    const std::vector<std::string> &get_symbols() const {
        return symbols_with_id->get_symbols();
    };

    /** Returns the cluster figures */
    const ClusterList &get_clusters() const {
        return clusters;
    };

    /** Return the cluster with the given name
     * The key in the map is the symmetry group
     */
    void get_clusters(const std::string &cname,
                      std::map<unsigned int, const Cluster *> &clusters) const;
    void get_clusters(const char *cname, std::map<unsigned int, const Cluster *> &clusters) const;

    /** Returns the translation matrix */
    const RowSparseStructMatrix &get_trans_matrix() const {
        return trans_matrix;
    };

    /** Get the translation symmetry group of a site */
    unsigned int get_trans_symm_group(unsigned int indx) const {
        return trans_symm_group[indx];
    };

    /** Sets the symbols */
    void set_symbols(const std::vector<std::string> &new_symbs);

    /** Set the number of threads to use during CF updating */
    void set_num_threads(unsigned int num) {
        cf_update_num_threads = num;
    };

    unsigned int get_num_threads() const {
        return this->cf_update_num_threads;
    }

    /** Find which sites have changed */
    void get_changes(const std::vector<std::string> &new_symbols,
                     std::vector<unsigned int> &changed_sites) const;

    /** Calculate correlation functions from scratch */
    void calculate_cf_from_scratch(const std::vector<std::string> &cf_names,
                                   std::map<std::string, double> &cf);

    /** Set a new atoms object */
    void set_atoms(PyObject *atoms);

    static void py_changes2_symb_changes(PyObject *all_changes,
                                         std::vector<SymbolChange> &symb_changes);

   private:
    void get_unique_indx_in_clusters(std::set<int> &unique_indx);

    /** Returns the maximum index occuring in the cluster indices */
    unsigned int get_max_indx_of_zero_site() const;

    unsigned int cf_update_num_threads{1};

    std::unique_ptr<Symbols> symbols_with_id{nullptr};

    ClusterList clusters;
    std::vector<int> trans_symm_group;
    std::vector<int> trans_symm_group_count;
    std::map<std::string, int> normalisation_factor;

    std::unique_ptr<BasisFunction> basis_functions{nullptr};

    Status_t status{Status_t::NOT_INITIALIZED};
    RowSparseStructMatrix trans_matrix;

    std::map<std::string, int> ctype_lookup;
    NamedArray eci;
    std::map<std::string, std::string> cname_with_dec;
    std::vector<bool> is_background_index;
    bool ignore_background_indices{true};
    bool assume_no_self_interactions{false};
    std::unique_ptr<CFHistoryTracker> history{nullptr};
    PyObject *atoms{nullptr};
    std::vector<std::string> singlets;
    // Pre-parsed names of the ECI values.
    std::vector<ParsedName> m_parsed_names;
    std::map<int, std::vector<ClusterCache>> m_cluster_by_symm;

    /* Prepare the next CF array */
    cf &get_next_cf(SymbolChange &symb_change);

    /** Compute the spin product for one element */
    double spin_product_one_atom(int ref_indx, const Cluster &cluster, const std::vector<int> &dec,
                                 int ref_id) const;
    // Calculate the change in spin product going from old_symb_id to new_symb_id
    double spin_product_one_atom_delta(const SpinProductCache &sp_cache, const Cluster &cluster,
                                       const deco_t &deco) const;

    /* Calculate the change in spin product under the assumption that no self-interaction
    is present, i.e. no site is present more than once in a figure within a cluster.*/
    double spin_product_one_atom_delta_no_si(const SpinProductCache &sp_cache,
                                             const Cluster &cluster, const deco_t &deco) const;

    /* Construct a map from the symmetry group with pointers to the relevant clusters.
    The pointer clusters are ordered in the same order as the ECI values.
    A nullpointer is inserted when a cluster is either a 0- or 1-body cluster, or not in the
    symmetry group.*/
    void cluster_by_symm_group();

    SpinProductCache build_sp_cache(const SymbolChange &symb_change) const;

    // Cached number of non background sites
    void count_non_bkg_sites();
    int num_non_bkg_sites;

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

    /* Pre-parse the ECI names for faster lookups */
    void parse_eci_names();
};
#endif
