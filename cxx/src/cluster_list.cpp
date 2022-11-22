#include "cluster_list.hpp"

void ClusterList::clear() {
    clusters.clear();
    symm_group_by_name.clear();
}

void ClusterList::append(const Cluster &cluster) {
    clusters[cluster.name].push_back(cluster);
    symm_group_by_name[cluster.name].insert(cluster.symm_group);
}

bool ClusterList::is_in_symm_group(const std::string &name, unsigned int symm_group) const {
    auto &group = symm_group_by_name.at(name);
    return group.find(symm_group) != group.end();
}

const Cluster &ClusterList::get(const std::string &name, unsigned int symm_group) {
    return find_cluster(name, symm_group);
}

Cluster *ClusterList::get_ptr(const std::string &name, unsigned int symm_group) {
    return &find_cluster(name, symm_group);
}

Cluster &ClusterList::find_cluster(const std::string &name, unsigned int symm_group) {
    for (Cluster &cluster : clusters.at(name)) {
        if (cluster.symm_group == symm_group) {
            return cluster;
        }
    }

    throw std::runtime_error("Did not find cluster in the requested symmetry group!");
}

unsigned int ClusterList::max_index() const {
    unsigned int mx = 0;
    for (auto iter = clusters.begin(); iter != clusters.end(); ++iter) {
        for (const Cluster &cluster : iter->second) {
            if (cluster.max_index() > mx) {
                mx = cluster.max_index();
            }
        }
    }
    return mx;
}

void ClusterList::unique_indices(std::set<int> &unique_indx) const {
    for (auto iter = clusters.begin(); iter != clusters.end(); ++iter) {
        for (const Cluster &cluster : iter->second) {
            cluster.unique_indices(unique_indx);
        }
    }
}
