#ifndef CLUSTER_H
#define CLUSTER_H
#include <Python.h>

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

typedef std::vector<int> deco_t;
typedef std::vector<std::vector<int>> cluster_t;
typedef std::vector<deco_t> equiv_deco_t;
typedef std::unordered_map<std::string, equiv_deco_t> all_equiv_deco_t;

struct ClusterSite {
   public:
    ClusterSite(unsigned int cluster_index, unsigned int lattice_index)
        : cluster_index(cluster_index), lattice_index(lattice_index){};
    unsigned int cluster_index;
    unsigned int lattice_index;
};
class Cluster {
   public:
    Cluster() : size(0), name("noname"){};
    Cluster(PyObject *info_dict);

    /** Returns the built in list */
    const cluster_t &get() const {
        return figures;
    };
    const cluster_t &get_order() const {
        return order;
    };
    const cluster_t &get_equiv() const {
        return equiv_sites;
    };
    const std::vector<double> &get_duplication_factors() const {
        return duplication_factors;
    };
    unsigned int get_size() const {
        return size;
    };

    inline unsigned int get_num_figures() const {
        return this->num_figures;
    };

    const equiv_deco_t &get_equiv_deco(const std::string &dec_string) const;
    const equiv_deco_t &get_equiv_deco(const std::vector<int> &deco) const;
    equiv_deco_t *get_equiv_deco_ptr(const std::string &dec_str);
    void unique_indices(std::set<int> &indices) const;

    /** Return the max index in the cluster */
    unsigned int max_index() const;

    /** Finds all the equivalent decoration numbers */
    void construct_equivalent_deco(int n_basis_funcs);
    /** Construct all available decoration numbers */
    std::vector<std::vector<int>> get_all_decoration_numbers(int n_basis_funcs) const;

    /** Public attributes */
    int size;
    std::string name;
    unsigned int ref_indx;
    unsigned int symm_group;
    double max_cluster_dia;

    const std::vector<int> &get_ref_cluster_sites() const {
        return this->ref_cluster_site;
    };

    const std::vector<ClusterSite> &get_non_ref_sites() const {
        return this->non_ref_sites;
    };

   private:
    cluster_t figures;
    cluster_t order;
    cluster_t equiv_sites;
    all_equiv_deco_t equiv_deco;
    std::vector<double> duplication_factors;

    std::vector<int> ref_cluster_site;
    std::vector<ClusterSite> non_ref_sites;
    unsigned int num_figures;

    void all_deco(int n_bfs, std::vector<std::vector<int>> &all_deco) const;

    static void deco2string(const std::vector<int> &deco, std::string &name);

    static void nested_list_to_cluster(PyObject *py_list, cluster_t &vec);

    /** Initialize figures based on the info dictionary */
    void parse_info_dict(PyObject *info_dict);

    /** Calculate additional factor due to duplications */
    void calculate_scaling_factors(PyObject *pylist);

    /** Check if the parsing went fine */
    void check_consistency() const;
};

std::ostream &operator<<(std::ostream &out, const Cluster &clust);
#endif
