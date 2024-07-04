#ifndef CLUSTER_LIST_H
#define CLUSTER_LIST_H
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "cluster.hpp"

class ClusterList {
   public:
    ClusterList(){};

    void clear();

    void append(const Cluster &cluster);

    /** Check if a cluster is in the symm group */
    bool is_in_symm_group(const std::string &name, unsigned int symm_group) const;

    /** Get a given cluster with name and symmetry group */
    const Cluster &get(const std::string &name, unsigned int symm_group);
    Cluster *get_ptr(const std::string &name, unsigned int symm_group);

    /** Return the maximum index present in any of the clusters */
    unsigned int max_index() const;

    void unique_indices(std::set<int> &indices) const;

   private:
    std::unordered_map<std::string, std::set<unsigned int>> symm_group_by_name;
    std::unordered_map<std::string, std::vector<Cluster>> clusters;

    Cluster &find_cluster(const std::string &name, unsigned int symm_group);
};

#endif
