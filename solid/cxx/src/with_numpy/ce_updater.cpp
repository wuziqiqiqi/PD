#include "ce_updater.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include "additional_tools.hpp"
#include "atomic_numbers.hpp"
#include "basis_function.hpp"
#include "cluster_name.hpp"
#include "config.hpp"

using namespace std;

CEUpdater::CEUpdater(){};
CEUpdater::~CEUpdater(){};

void CEUpdater::init(PyObject *py_atoms, PyObject *settings, PyObject *corrFunc, PyObject *pyeci,
                     PyObject *cluster_list) {
    atoms = py_atoms;
    if (settings == nullptr) {
        throw invalid_argument("Settings object is nullptr!");
    }

#ifdef PRINT_DEBUG
    cout << "Getting symbols from settings object\n";
#endif
    PyObject *py_ignore_bck = get_attr(settings, "ignore_background_atoms");
    ignore_background_indices = PyObject_IsTrue(py_ignore_bck);
    Py_DECREF(py_ignore_bck);

    unsigned int n_atoms = PyObject_Length(atoms);
    if (n_atoms < 0) {
        throw invalid_argument("Could not retrieve the length of the atoms object!");
    }

    // Read the atomic symbols
    std::vector<std::string> symbols = get_symbols_from_atoms(py_atoms);

    trans_symm_group.resize(n_atoms);
    set<string> unique_symbols;

    // Extract unique symbols from settings
    PyObject *py_unique_symb = get_attr(settings, "unique_elements");
    for (unsigned int i = 0; i < list_size(py_unique_symb); i++) {
        unique_symbols.insert(py2string(PyList_GetItem(py_unique_symb, i)));
    }
    Py_DECREF(py_unique_symb);

    insert_in_set(symbols, unique_symbols);
    symbols_with_id = std::make_unique<Symbols>(symbols, unique_symbols);

    // Build read the translational sites
    PyObject *py_trans_symm_group = get_attr(settings, "index_by_sublattice");

    if (py_trans_symm_group == nullptr) {
        throw invalid_argument("index_by_sublattice is nullptr!");
    }

#ifdef PRINT_DEBUG
    cout << "Reading background indices\n";
#endif
    // Read the backgound indices from settings
    PyObject *bkg_indx = get_attr(settings, "background_indices");
    read_background_indices(bkg_indx);
    Py_DECREF(bkg_indx);

    count_non_bkg_sites();
    build_trans_symm_group(py_trans_symm_group);
    Py_DECREF(py_trans_symm_group);

#ifdef PRINT_DEBUG
    cout << "Getting cluster names from atoms object\n";
#endif

    // Read cluster names
    create_cname_with_dec(corrFunc);

#ifdef PRINT_DEBUG
    cout << "Cluster names with decoration number created...\n";
#endif

    PyObject *py_num_elements = get_attr(settings, "num_unique_elements");

    if (py_num_elements == nullptr) {
        throw invalid_argument("num_unique_elements is nullptr!");
    }
    int num_bfs = py2int(py_num_elements) - 1;
    Py_DECREF(py_num_elements);

    if (cluster_list == nullptr) {
        throw invalid_argument("cluster_list is nullptr!");
    }
    // unsigned int num_trans_symm = list_size(cluster_info);
    unsigned int num_clusters = PySequence_Size(cluster_list);

#ifdef PRINT_DEBUG
    cout << "Parsing cluster list...\n";
#endif

    PyObject *py_no_si = get_attr(cluster_list, "assume_no_self_interactions");
    assume_no_self_interactions = PyObject_IsTrue(py_no_si);
    Py_DECREF(py_no_si);

#ifdef PRINT_DEBUG
    std::cout << "Assuming no self-interaction?: " << assume_no_self_interactions << std::endl;
#endif

    for (unsigned int i = 0; i < num_clusters; i++) {
        PyObject *py_cluster = PySequence_GetItem(cluster_list, i);

        Cluster new_clst(py_cluster);
        PyObject *py_cluster_name = get_attr(py_cluster, "name");
        string cluster_name = py2string(py_cluster_name);
        Py_DECREF(py_cluster_name);
        Py_DECREF(py_cluster);

        new_clst.construct_equivalent_deco(num_bfs);
        clusters.append(new_clst);

        int norm_fact = new_clst.get().size() * trans_symm_group_count[new_clst.symm_group];
        if (normalisation_factor.find(cluster_name) == normalisation_factor.end()) {
            normalisation_factor[cluster_name] = norm_fact;
        } else {
            normalisation_factor[cluster_name] += norm_fact;
        }
    }
#ifdef PRINT_DEBUG
    cout << "Finished reading cluster_info\n";
#endif

#ifdef PRINT_DEBUG
    cout << "Reading basis functions from settings object\n";
#endif

    PyObject *bfs = get_attr(settings, "basis_functions");
    if (bfs == NULL) {
        status = Status_t::INIT_FAILED;
        return;
    }

    // Reading basis functions from python object
    PyObject *key;
    PyObject *value;
    unsigned int n_bfs = list_size(bfs);
    bf_list basis_func_raw;
    for (unsigned int i = 0; i < n_bfs; i++) {
        Py_ssize_t pos = 0;
        map<string, double> new_entry;
        PyObject *bf_dict = PyList_GetItem(bfs, i);
        while (PyDict_Next(bf_dict, &pos, &key, &value)) {
            new_entry[py2string(key)] = PyFloat_AS_DOUBLE(value);
        }
        basis_func_raw.push_back(new_entry);
    }

    this->basis_functions = std::make_unique<BasisFunction>(basis_func_raw, *symbols_with_id);

#ifdef PRINT_DEBUG
    cout << "Reading translation matrix from settings\n";
#endif
    // Retrieve the TransMatrix object
    PyObject *trans_mat_obj = get_attr(settings, "trans_matrix");
    if (trans_mat_obj == NULL) {
        status = Status_t::INIT_FAILED;
        return;
    }

    // Get the internal trans_matrix object from within the TransMatrix object
    PyObject *trans_mat_orig = get_attr(trans_mat_obj, "trans_matrix");

    read_trans_matrix(trans_mat_orig);
    Py_DECREF(trans_mat_obj);
    Py_DECREF(trans_mat_orig);

    // Read the ECIs, and parse the names.
    this->set_eci(pyeci);
#ifdef PRINT_DEBUG
    cout << "Parsing correlation function\n";
#endif

    vector<string> flattened_cnames;
    flattened_cf_names(flattened_cnames);

    history = std::make_unique<CFHistoryTracker>(eci.get_names());
    history->insert(corrFunc, nullptr);

    // Store the singlets names
    for (unsigned int i = 0; i < flattened_cnames.size(); i++) {
        std::string name = flattened_cnames[i];
        // Fetch the pre-parsed version of the name.
        const ParsedName parsed = this->m_parsed_names[i];
        if (parsed.size == 1) {
            singlets.push_back(name);
        }
    }

    status = Status_t::READY;
    clear_history();
#ifdef PRINT_DEBUG
    cout << "CEUpdater initialized sucessfully!\n";
#endif

    // Verify that the ECIs given corresponds to a correlation function
    if (!all_eci_corresponds_to_cf()) {
        throw invalid_argument("All ECIs does not correspond to a correlation function!");
    }
}

void CEUpdater::parse_eci_names() {
    std::size_t num = eci.size();
    this->m_parsed_names.clear();
    this->m_parsed_names.reserve(num);

    for (unsigned int i = 0; i < num; i++) {
        std::string name = eci.name(i);
        ClusterName c_name = ClusterName(name);
        ParsedName parsed = c_name.get_parsed();
        this->m_parsed_names.emplace_back(parsed);
    }
}

double CEUpdater::get_energy() {
    double energy = 0.0;
    cf &corr_func = history->get_current();
    energy = eci.dot(corr_func);
    return energy * symbols_with_id->size();
}

double CEUpdater::spin_product_one_atom(int ref_indx, const Cluster &cluster,
                                        const vector<int> &dec, int ref_id) const {
    double sp = 0.0;

    // Note: No duplication factor is included here, since this method is used for calculating the
    // CF from scratch (which will account for self-interactions with no issue), and not updating
    // (which must account for self-interactions via the duplication factor).
    // i.e. spin_product_one_atom_delta must accounts for the self-interaction.

    const vector<vector<int>> &indx_list = cluster.get();
    unsigned int num_indx = indx_list.size();
    unsigned int n_memb = indx_list[0].size();

    // Cache the relevant row from the trans matrix.
    int *trans_matrix_row = trans_matrix.get_row(ref_indx);

    for (unsigned int i = 0; i < num_indx; i++) {
        double sp_temp = 1.0;

        // Use pointer arithmetics in the inner most loop
        const int *indices = &indx_list[i][0];

        for (unsigned int j = 0; j < n_memb; j++) {
            int trans_index = trans_matrix.lookup_in_row(trans_matrix_row, indices[j]);
            int id = (trans_index == ref_indx) ? ref_id : symbols_with_id->id(trans_index);
            sp_temp *= basis_functions->get(dec[j], id);
        }
        sp += sp_temp;
    }
    return sp;
}

int CEUpdater::get_original_index(int ref_indx) const {
    int *trans_matrix_row = trans_matrix.get_row(ref_indx);
    int *allowed_lu = trans_matrix.get_allowed_lookup_values();
    for (unsigned int j = 0; j < trans_matrix.get_num_non_zero(); j++) {
        int col = allowed_lu[j];
        int indx = trans_matrix.lookup_in_row(trans_matrix_row, col);
        if (indx == ref_indx) {
            return col;
        }
    }
    std::stringstream err;
    err << "Did not locate original index for ref index: " << ref_indx;
    throw std::runtime_error(err.str());
}

double CEUpdater::spin_product_one_atom_delta_no_si(const SpinProductCache &sp_cache,
                                                    const Cluster &cluster,
                                                    const deco_t &deco) const {
    /* Note: This function assumes no self-interaction within a cluster. */

    // Figure out how many times we need to iterate
    unsigned int num_indx = cluster.get_num_figures();  // Outer loop count
    // Assume 1 ref site in a figure, so we iterate 1 less
    unsigned int n_non_ref = cluster.get_size() - 1;  // Inner loop count

    int *tm_row = sp_cache.trans_matrix_row;

    /* There are n_non_ref sites per each ref site, so the non_ref_site_ptr
    iterates faster than the ref_site_ptr. Figures are placed contiguously
    in a 1D array.*/
    const ClusterSite *non_ref_site_ptr = &cluster.get_non_ref_sites()[0];
    const int *ref_site_ptr = &cluster.get_ref_cluster_sites()[0];

    // Keep track of the change in spin-product
    double sp_delta = 0.0;

    // Iterate each figure in the cluster. 1 reference site is assumed per cluster
    for (unsigned int i = 0; i < num_indx; i++, ++ref_site_ptr) {
        /* Calculate the spin product for both new and the old (ref)
        The constant term to the spin product from the sites which didn't change.*/
        const int dec_ref = deco[*ref_site_ptr];
        double new_bf = basis_functions->get(dec_ref, sp_cache.new_symb_id);
        double old_bf = basis_functions->get(dec_ref, sp_cache.old_symb_id);
        double sp_change = new_bf - old_bf;

        /* Iterate the remaining non-reference sites, as we already took care of the reference
        site (assuming no self-interaction)*/
        for (unsigned int j = 0; j < n_non_ref; j++, ++non_ref_site_ptr) {
            const ClusterSite &site = *non_ref_site_ptr;
            const int dec_j = deco[site.cluster_index];

            const int trans_index = trans_matrix.lookup_in_row(tm_row, site.lattice_index);
            sp_change *= basis_functions->get(dec_j, symbols_with_id->id(trans_index));
        }

        sp_delta += sp_change;
    }
    return sp_delta;
}

double CEUpdater::spin_product_one_atom_delta(const SpinProductCache &sp_cache,
                                              const Cluster &cluster, const deco_t &deco) const {
    // Keep track of the change in spin-product
    double sp_delta = 0.0;

    // List of figures in the cluster
    const vector<vector<int>> &indx_list = cluster.get();
    // Account for the self-interaction, in case we updated 2 sites with 1 change
    const std::vector<double> &dup_factors = cluster.get_duplication_factors();
    unsigned int num_indx = indx_list.size();
    unsigned int n_memb = indx_list[0].size();

    int *tm_row = sp_cache.trans_matrix_row;

    // Iterate each site in the cluster
    for (unsigned int i = 0; i < num_indx; i++) {
        // Use pointer arithmetics in the inner most loop
        const int *indices = &indx_list[i][0];

        // Calculate the spin product for both new and the old (ref)
        double sp_temp_new = 1.0, sp_temp_ref = 1.0;
        // The constant term to the spin product from the sites which didn't change.
        double sp_const = dup_factors[i];

        for (unsigned int j = 0; j < n_memb; j++) {
            const int site_index = indices[j];
            const int dec_j = deco[j];
            if (site_index == sp_cache.original_index) {
                // This site is the reference index.
                // Look up the BF values directly for the new and old symbols
                sp_temp_new *= basis_functions->get(dec_j, sp_cache.new_symb_id);
                sp_temp_ref *= basis_functions->get(dec_j, sp_cache.old_symb_id);
            } else {
                // Look up the symbol of the non-reference site, which hasn't changed.
                const int trans_index = trans_matrix.lookup_in_row(tm_row, site_index);
                sp_const *= basis_functions->get(dec_j, symbols_with_id->id(trans_index));
            }
        }
        // The change in spin-product is the difference in SP between the site(s) which
        // changed, multiplied by the constant SP from the other un-changed sites (since
        // these contribute equally before and after the change).
        sp_delta += (sp_temp_new - sp_temp_ref) * sp_const;
    }
    return sp_delta;
}

void CEUpdater::update_cf(PyObject *single_change) {
    SymbolChange symb_change = SymbolChange(single_change);
    update_cf(symb_change);
}

void CEUpdater::py_changes2_symb_changes(PyObject *all_changes,
                                         vector<SymbolChange> &symb_changes) {
    unsigned int size = list_size(all_changes);
    for (unsigned int i = 0; i < size; i++) {
        SymbolChange symb_change = SymbolChange(PyList_GetItem(all_changes, i));
        symb_changes.push_back(symb_change);
    }
}

SpinProductCache CEUpdater::build_sp_cache(const SymbolChange &symb_change) const {
    int ref_indx = symb_change.indx;
    // Look up the untranslated index of the reference index.
    int orig_indx = this->get_original_index(ref_indx);
    // Cache the relevant row from the trans matrix.
    int *trans_matrix_row = this->trans_matrix.get_row(ref_indx);

    unsigned int old_symb_id = symbols_with_id->get_symbol_id(symb_change.old_symb);
    unsigned int new_symb_id = symbols_with_id->get_symbol_id(symb_change.new_symb);

    SpinProductCache sp_cache = {ref_indx, orig_indx, trans_matrix_row, new_symb_id, old_symb_id};
    return sp_cache;
}

cf &CEUpdater::get_next_cf(SymbolChange &symb_change) {
    SymbolChange *symb_change_track;
    cf *next_cf_ptr = nullptr;
    history->get_next(&next_cf_ptr, &symb_change_track);
    cf &next_cf = *next_cf_ptr;

    symb_change_track->indx = symb_change.indx;
    symb_change_track->old_symb = symb_change.old_symb;
    symb_change_track->new_symb = symb_change.new_symb;
    symb_change_track->track_indx = symb_change.track_indx;
    return next_cf;
}

void CEUpdater::update_cf(SymbolChange &symb_change) {
    if (symb_change.old_symb == symb_change.new_symb) {
        return;
    }

    if (is_background_index[symb_change.indx]) {
        throw runtime_error("Attempting to move a background atom!");
    }

    cf &current_cf = history->get_current();
    cf &next_cf = get_next_cf(symb_change);

    symbols_with_id->set_symbol(symb_change.indx, symb_change.new_symb);

    /* The following prepares a range of properties which will be
     useful for all of the clusters, so we don't compute more
     than we have to inside the main ECI loop */
    SpinProductCache sp_cache = this->build_sp_cache(symb_change);

    if (atoms != nullptr) {
        set_symbol_in_atoms(atoms, symb_change.indx, symb_change.new_symb);
    }

    int symm = trans_symm_group[symb_change.indx];
    const std::vector<ClusterCache> &clusters_cache = m_cluster_by_symm[symm];

    // Loop over all ECIs
    // As work load for different clusters are different due to a different
    // multiplicity factor, we need to apply a dynamic schedule
#ifdef HAS_OMP
    bool is_par = this->cf_update_num_threads > 1;
#pragma omp parallel for if (is_par) num_threads(this->cf_update_num_threads) schedule(dynamic)
#endif
    for (unsigned int i = 0; i < eci.size(); i++) {
        // The pre-parsed version of the cluster name.
        const ParsedName &parsed = this->m_parsed_names[i];

        // 0-body
        if (parsed.size == 0) {
            next_cf[i] = current_cf[i];
            continue;
        }

        // Singlet
        if (parsed.size == 1) {
            unsigned int dec = parsed.dec_num;
            double new_bf = basis_functions->get(dec, sp_cache.new_symb_id);
            double old_bf = basis_functions->get(dec, sp_cache.old_symb_id);
            next_cf[i] = current_cf[i] + (new_bf - old_bf) / num_non_bkg_sites;
            continue;
        }

        // n-body
        const ClusterCache &cluster_cache = clusters_cache[i];
        const Cluster *cluster_ptr = cluster_cache.cluster_ptr;

        if (cluster_ptr == nullptr) {
            // This cluster was not present in the symmetry group.
            next_cf[i] = current_cf[i];
            continue;
        }
        // The cluster is in the symmetry group, so calculate the spin product
        // change for this cluster.
        const Cluster &cluster = *cluster_ptr;
        const equiv_deco_t &equiv_deco = *cluster_cache.equiv_deco_ptr;

        double delta_sp = 0.0;
        // Calculate the change (delta) in spin product
        for (const deco_t &deco : equiv_deco) {
            if (this->assume_no_self_interactions) {
                // Faster version for large cells with no self interaction
                delta_sp += spin_product_one_atom_delta_no_si(sp_cache, cluster, deco);
            } else {
                // Safe fall-back version
                delta_sp += spin_product_one_atom_delta(sp_cache, cluster, deco);
            }
        }

        delta_sp *= cluster_cache.normalization;
        next_cf[i] = current_cf[i] + delta_sp;
    }
}

void CEUpdater::undo_changes() {
    unsigned int buf_size = history->history_size();
    undo_changes(buf_size - 1);
}

void CEUpdater::undo_changes(int num_steps) {
    int buf_size = history->history_size();

    if (num_steps > buf_size - 1) {
        throw invalid_argument("Can't reset history beyond the buffer size!");
    }

    SymbolChange *last_changes;
    for (int i = 0; i < num_steps; i++) {
        history->pop(&last_changes);
        symbols_with_id->set_symbol(last_changes->indx, last_changes->old_symb);

        if (atoms != nullptr) {
            set_symbol_in_atoms(atoms, last_changes->indx, last_changes->old_symb);
        }
    }
}

double CEUpdater::calculate(PyObject *system_changes) {
    unsigned int size = list_size(system_changes);
    if (size == 0) {
        return get_energy();
    } else if (size == 1) {
        for (unsigned int i = 0; i < size; i++) {
            update_cf(PyList_GetItem(system_changes, i));
        }
        return get_energy();
    }

    if (size % 2 == 0) {
        bool sequence_arbitrary_moves = false;
        vector<swap_move> sequence;
        for (unsigned int i = 0; i < size / 2; i++) {
            swap_move changes;
            changes[0] = SymbolChange(PyList_GetItem(system_changes, 2 * i));
            changes[1] = SymbolChange(PyList_GetItem(system_changes, 2 * i + 1));

            if (!is_swap_move(changes)) {
                sequence_arbitrary_moves = true;
                break;
            }
            sequence.push_back(changes);
        }

        if (!sequence_arbitrary_moves) {
            return calculate(sequence);
        }
    }

    // Last option is that this is a sequence of arbitrary moves
    vector<SymbolChange> changes(size);
    for (unsigned int i = 0; i < size; i++) {
        changes[i] = SymbolChange(PyList_GetItem(system_changes, i));
    }
    return calculate(changes);
}

double CEUpdater::calculate(swap_move &system_changes) {
    if (symbols_with_id->id(system_changes[0].indx) ==
        symbols_with_id->id(system_changes[1].indx)) {
        cout << system_changes[0] << endl;
        cout << system_changes[1] << endl;
        throw runtime_error(
            "This version of the calculate function assumes that the provided update is swapping "
            "two atoms\n");
    }

    if (symbols_with_id->get_symbol(system_changes[0].indx) != system_changes[0].old_symb) {
        throw runtime_error("The atom position tracker does not match the current state\n");
    } else if (symbols_with_id->get_symbol(system_changes[1].indx) != system_changes[1].old_symb) {
        throw runtime_error("The atom position tracker does not match the current state\n");
    }

    // Update correlation function
    update_cf(system_changes[0]);
    update_cf(system_changes[1]);

    return get_energy();
}

void CEUpdater::clear_history() {
    history->clear();
}

void CEUpdater::flattened_cf_names(vector<string> &flattened) {
    flattened = eci.get_names();

    // Sort the cluster names for consistency
    sort(flattened.begin(), flattened.end());
}

PyObject *CEUpdater::get_cf() {
    PyObject *cf_dict = PyDict_New();
    cf &corrfunc = history->get_current();

    for (unsigned int i = 0; i < corrfunc.size(); i++) {
        PyObject *pyvalue = PyFloat_FromDouble(corrfunc[i]);
        PyDict_SetItemString(cf_dict, corrfunc.name(i).c_str(), pyvalue);
        Py_DECREF(pyvalue);
    }
    return cf_dict;
}

void CEUpdater::set_symbols(const vector<string> &new_symbs) {
    if (new_symbs.size() != symbols_with_id->size()) {
        throw runtime_error(
            "The number of atoms in the updater cannot be changed via the set_symbols function\n");
    }
    symbols_with_id->set_symbols(new_symbs);
}

void CEUpdater::cluster_by_symm_group() {
    m_cluster_by_symm.clear();

    // Find unique symmetry groups
    std::set<int> unique;
    insert_in_set(this->trans_symm_group, unique);

    for (const int symm : unique) {
        if (symm == -1) {
            // Background symmetry group
            continue;
        }
        // 1 ClusterCache per ECI value
        std::vector<ClusterCache> cluster_cache;
        cluster_cache.reserve(this->m_parsed_names.size());
        for (const ParsedName &parsed : this->m_parsed_names) {
            ClusterCache cache;
            if (parsed.size == 0 || parsed.size == 1 ||
                !clusters.is_in_symm_group(parsed.prefix, symm)) {
                /* Either 0- or 1-body cluster, or cluster is not in this
                symmetry group. */
                cluster_cache.push_back(cache);
                continue;
            }
            Cluster *cluster_ptr = clusters.get_ptr(parsed.prefix, symm);
            equiv_deco_t *equiv_ptr = cluster_ptr->get_equiv_deco_ptr(parsed.dec_str);
            // Calculate the normalization of the resulting cluster functions
            double normalization = static_cast<double>(cluster_ptr->size);
            normalization /= equiv_ptr->size();
            normalization /= normalisation_factor.at(parsed.prefix);

            // Populate the new cache object
            cache.cluster_ptr = cluster_ptr;
            cache.equiv_deco_ptr = equiv_ptr;
            cache.normalization = normalization;

            cluster_cache.push_back(cache);
        }
        m_cluster_by_symm.insert({symm, cluster_cache});
    }
}
void CEUpdater::set_eci(PyObject *pyeci) {
    PyObject *key;
    PyObject *value;

    // Read the ECIs
    Py_ssize_t pos = 0;
    std::map<std::string, double> temp_eci;

    while (PyDict_Next(pyeci, &pos, &key, &value)) {
        temp_eci[py2string(key)] = PyFloat_AS_DOUBLE(value);
    }
    this->eci.init(temp_eci);
    // Pre-parse the names of the clusters.
    this->parse_eci_names();

    // If status is not READY, then we're still initializing, and CF's are missing.
    if (this->status == Status_t::READY && !all_eci_corresponds_to_cf()) {
        throw invalid_argument("All ECIs has to correspond to a correlation function!");
    }
    // Update the cluster pointers to match the order with the ECI's.
    cluster_by_symm_group();
}

bool CEUpdater::all_decoration_nums_equal(const vector<int> &dec_nums) const {
    for (unsigned int i = 1; i < dec_nums.size(); i++) {
        if (dec_nums[i] != dec_nums[0]) {
            return false;
        }
    }
    return true;
}

void CEUpdater::get_singlets(PyObject *npy_obj) const {
    PyObject *npy_array = PyArray_FROM_OTF(npy_obj, NPY_DOUBLE, NPY_OUT_ARRAY);
    unsigned int npy_array_size = PyArray_SIZE(npy_array);
    if (npy_array_size < singlets.size()) {
        string msg("The passed Numpy array is too small to hold all the singlets terms!\n");
        stringstream ss;
        ss << "Minimum size: " << singlets.size() << ". Given size: " << npy_array_size;
        msg += ss.str();
        Py_DECREF(npy_array);
        throw runtime_error(msg);
    }
    cf &cfs = history->get_current();
    for (unsigned int i = 0; i < singlets.size(); i++) {
        double *ptr = static_cast<double *>(PyArray_GETPTR1(npy_array, i));
        *ptr = cfs[singlets[i]];
    }
    Py_DECREF(npy_array);
}

PyObject *CEUpdater::get_singlets() const {
    npy_intp dims[1] = {static_cast<npy_intp>(singlets.size())};
    PyObject *npy_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    get_singlets(npy_array);
    return npy_array;
}

void CEUpdater::create_cname_with_dec(PyObject *cf) {
    if (!PyDict_Check(cf)) {
        throw invalid_argument("Correlation functons has to be dictionary!");
    }
    Py_ssize_t pos = 0;
    PyObject *key;
    PyObject *value;
    while (PyDict_Next(cf, &pos, &key, &value)) {
        string new_key = py2string(key);
        ClusterName c_name = ClusterName(new_key);
        unsigned int size = c_name.get_size();
#ifdef PRINT_DEBUG
        cout << "Read CF: " << new_key << endl;
#endif
        if ((size == 0 || size == 1)) {
            cname_with_dec[new_key] = new_key;
        } else {
            std::string prefix = c_name.get_prefix();
            cname_with_dec[prefix] = new_key;
        }
    }
}

void CEUpdater::build_trans_symm_group(PyObject *py_trans_symm_group) {
    // Fill the symmetry group array with -1 indicating an invalid value
    for (unsigned int i = 0; i < trans_symm_group.size(); i++) {
        trans_symm_group[i] = -1;
    }

    unsigned int py_list_size = list_size(py_trans_symm_group);
    for (unsigned int i = 0; i < py_list_size; i++) {
        PyObject *sublist = PyList_GetItem(py_trans_symm_group, i);
        unsigned int n_sites = list_size(sublist);
        for (unsigned int j = 0; j < n_sites; j++) {
            int indx = py2int(PyList_GetItem(sublist, j));
            if (trans_symm_group[indx] != -1) {
                throw runtime_error(
                    "One site appears to be present in more than one translation symmetry "
                    "group!");
            }
            trans_symm_group[indx] = i;
        }
    }

    // Check that all sites belongs to one translational symmetry group
    for (unsigned int i = 0; i < trans_symm_group.size(); i++) {
        if ((trans_symm_group[i] == -1) && !is_background_index[i]) {
            stringstream msg;
            msg << "Site " << i << " has not been assigned to any translational symmetry group!";
            throw runtime_error(msg.str());
        }
    }

    // Count the number of atoms in each symmetry group
    trans_symm_group_count.resize(py_list_size);
    fill(trans_symm_group_count.begin(), trans_symm_group_count.end(), 0);
    for (unsigned int i = 0; i < trans_symm_group.size(); i++) {
        if (trans_symm_group[i] >= 0) {
            trans_symm_group_count[trans_symm_group[i]] += 1;
        }
    }
}

bool CEUpdater::all_eci_corresponds_to_cf() {
    cf &corrfunc = history->get_current();
    return eci.names_are_equal(corrfunc);
}

double CEUpdater::calculate(vector<swap_move> &sequence) {
    if (sequence.size() >= history->max_history / 2) {
        throw invalid_argument(
            "The length of sequence of swap move exceeds the buffer size for the history "
            "tracker");
    }

    for (unsigned int i = 0; i < sequence.size(); i++) {
        calculate(sequence[i]);
    }
    return get_energy();
}

double CEUpdater::calculate(vector<SymbolChange> &sequence) {
    for (auto &change : sequence) {
        update_cf(change);
    }
    return get_energy();
}

void CEUpdater::read_trans_matrix(PyObject *py_trans_mat) {
    bool is_list = PyList_Check(py_trans_mat);

#ifdef PRINT_DEBUG
    cout << "read_trans_matrix: Extracting unique indices" << endl;
#endif

    set<int> unique_indx;
    clusters.unique_indices(unique_indx);
    vector<int> unique_indx_vec;
    set2vector(unique_indx, unique_indx_vec);

    // Compute the max index that is ever going to be checked
    unsigned int max_indx = clusters.max_index();
    if (is_list) {
        unsigned int size = list_size(py_trans_mat);

#ifdef PRINT_DEBUG
        cout << "read_trans_matrix: Updating size of trans_matrix" << endl;
#endif

        trans_matrix.set_size(size, unique_indx_vec.size(), max_indx);

#ifdef PRINT_DEBUG
        cout << "read_trans_matrix: Setting lookup values in trans_matrix" << endl;
#endif
        trans_matrix.set_lookup_values(unique_indx_vec);

#ifdef PRINT_DEBUG
        cout << "read_trans_matrix: Reading translation matrix from list of dictionaries" << endl;
#endif
        unsigned int n_elements_insterted = 0;
        for (unsigned int i = 0; i < size; i++) {
            // Background atoms are ignored (and should never be accessed)
            if (is_background_index[i] && ignore_background_indices) {
                continue;
            }

            PyObject *dict = PyList_GetItem(py_trans_mat, i);
            for (unsigned int j = 0; j < unique_indx_vec.size(); j++) {
                int col = unique_indx_vec[j];
                PyObject *value = PyDict_GetItem(dict, int2py(col));

                if (value == NULL) {
                    stringstream ss;
                    ss << "Requested value " << col << " is not a key in the dictionary!";
                    throw invalid_argument(ss.str());
                }
                trans_matrix(i, col) = py2int(value);
                n_elements_insterted++;
            }
        }
#ifdef PRINT_DEBUG
        cout << "Inserted " << n_elements_insterted << " into the translation matrix\n";
#endif
    } else {
        PyObject *trans_mat = PyArray_FROM_OTF(py_trans_mat, NPY_INT32, NPY_ARRAY_IN_ARRAY);

        npy_intp *size = PyArray_DIMS(trans_mat);
        trans_matrix.set_size(size[0], unique_indx_vec.size(), max_indx);
        trans_matrix.set_lookup_values(unique_indx_vec);

#ifdef PRINT_DEBUG
        cout << "Dimension of translation matrix stored: " << size[0] << " "
             << unique_indx_vec.size() << endl;
#endif

        if (max_indx + 1 > size[1]) {
            stringstream ss;
            ss << "Something is wrong with the translation matrix passed.\n";
            ss << "Shape of translation matrix (" << size[0] << "," << size[1] << ")\n";
            ss << "Maximum index encountered in the cluster lists: " << max_indx << endl;
            throw invalid_argument(ss.str());
        }
        for (unsigned int i = 0; i < size[0]; i++)
            for (unsigned int j = 0; j < unique_indx_vec.size(); j++) {
                int col = unique_indx_vec[j];
                trans_matrix(i, col) = *static_cast<int *>(PyArray_GETPTR2(trans_mat, i, col));
            }
        Py_DECREF(trans_mat);
    }
}

void CEUpdater::sort_indices(int indices[], const vector<int> &order, unsigned int n_indices) {
    // This function is called many times
    // profiling (with YEP) revealed that
    // [] operator of the vector used quite a bit of time
    // Therefore we here use raw C-arrays or pointer arithmetics
    int sorted[4];
    const int *ptr = &order[0];
    for (unsigned int i = 0; i < n_indices; i++) {
        sorted[i] = indices[*(ptr + i)];
    }
    memcpy(indices, sorted, n_indices * sizeof(int));
}

bool CEUpdater::is_swap_move(const swap_move &move) const {
    return (move[0].old_symb == move[1].new_symb) && (move[1].new_symb == move[1].old_symb);
}

void CEUpdater::read_background_indices(PyObject *bkg_indices) {
    // Fill array with false
    is_background_index.resize(symbols_with_id->size());
    fill(is_background_index.begin(), is_background_index.end(), false);

    // Set to true if index is in bkg_indices
    int size = list_size(bkg_indices);
    for (int i = 0; i < size; i++) {
        PyObject *py_indx = PyList_GetItem(bkg_indices, i);
        int indx = py2int(py_indx);
        is_background_index[indx] = true;
    }
}

void CEUpdater::count_non_bkg_sites() {
    // Count and store the number of non-background sites
    num_non_bkg_sites = 0;
    for (unsigned int atom_no = 0; atom_no < symbols_with_id->size(); atom_no++) {
        if (!is_background_index[atom_no] || !ignore_background_indices) {
            num_non_bkg_sites += 1;
        }
    }
}

void CEUpdater::get_changes(const std::vector<std::string> &new_symbols,
                            std::vector<unsigned int> &changed_sites) const {
    if (new_symbols.size() != symbols_with_id->size()) {
        throw invalid_argument("Size of passed atoms does not match!");
    }

    for (unsigned int i = 0; i < new_symbols.size(); i++) {
        unsigned int symb_id = symbols_with_id->id(i);
        if (symbols_with_id->get_symbol_id(new_symbols[i]) != symb_id) {
            changed_sites.push_back(i);
        }
    }
}

void CEUpdater::calculate_cf_from_scratch(const vector<string> &cf_names, map<string, double> &cf) {
    cf.clear();

    // Initialise all cluster names
    for (const string &name : cf_names) {
        cf[name] = 0.0;
    }

    // Loop over all clusters
    for (const string &name : cf_names) {
        ClusterName c_name = ClusterName(name);
        unsigned int cluster_size = c_name.get_size();

        // Handle empty cluster
        if (cluster_size == 0) {
            cf[name] = 1.0;
            continue;
        }

        // Handle singlet cluster
        if (cluster_size == 1) {
            unsigned int dec = c_name.get_dec_num();
            double new_value = 0.0;
            // Normalise with respect to the actual number of atoms included
            for (unsigned int atom_no = 0; atom_no < symbols_with_id->size(); atom_no++) {
                if (!is_background_index[atom_no] || !ignore_background_indices) {
                    new_value += basis_functions->get(dec, symbols_with_id->id(atom_no));
                }
            }
            cf[name] = new_value / num_non_bkg_sites;
            continue;
        }

        // Handle the rest of the clusters
        std::string prefix, dec_str;
        c_name.get_prefix_and_dec_str(prefix, dec_str);

        double sp = 0.0;
        double count = 0;
        for (unsigned int atom_no = 0; atom_no < symbols_with_id->size(); atom_no++) {
            int symm = trans_symm_group[atom_no];

            if ((!clusters.is_in_symm_group(prefix, symm)) ||
                (is_background_index[atom_no] && ignore_background_indices)) {
                continue;
            }

            const Cluster &cluster = clusters.get(prefix, symm);

            const equiv_deco_t &equiv_deco = cluster.get_equiv_deco(dec_str);
            unsigned int ref_id = symbols_with_id->id(atom_no);
            double sp_temp = 0.0;
            for (const vector<int> &deco : equiv_deco) {
                sp_temp += spin_product_one_atom(atom_no, cluster, deco, ref_id);
            }
            sp += sp_temp / equiv_deco.size();
            count += cluster.get().size();
        }

        if (count == 0) {
            cf[name] = 0.0;
        } else {
            cf[name] = sp / count;
        }
    }

    history->get_current().init(cf);
}

void CEUpdater::set_atoms(PyObject *py_atoms) {
    unsigned int num_atoms = PySequence_Length(py_atoms);

    if (num_atoms != symbols_with_id->size()) {
        throw invalid_argument("Length of passed atoms object is different from current");
    }

    std::vector<std::string> symbols = get_symbols_from_atoms(py_atoms);

    this->atoms = py_atoms;
    symbols_with_id->set_symbols(symbols);
}
