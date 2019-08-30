#include "cluster_list.hpp"

using namespace std;


void ClusterList::clear(){
    clusters.clear();
    symm_group_by_name.clear();
}

void ClusterList::append(const Cluster &cluster){
    clusters[cluster.name].push_back(cluster);
    symm_group_by_name[cluster.name].insert(cluster.symm_group);
}

bool ClusterList::is_in_symm_group(const string &name, unsigned int symm_group) const{
    return symm_group_by_name.at(name).find(symm_group) != symm_group_by_name.at(name).end();
}

const Cluster& ClusterList::get(const string &name, unsigned int symm_group) const{
    for (const Cluster& cluster: clusters.at(name)){
        if (cluster.symm_group == symm_group){
            return cluster;
        }
    }

    throw runtime_error("Did not find cluster in the requested symmetry group!");
}

unsigned int ClusterList::max_index() const{
    unsigned int mx = 0;
    for (auto iter=clusters.begin(); iter != clusters.end(); ++iter){
        for (const Cluster& cluster: iter->second){
            if (cluster.max_index() > mx){
                mx = cluster.max_index();
            }
        }
    }
    return mx;
}

void ClusterList::unique_indices(set<int> &unique_indx) const{
    for (auto iter=clusters.begin(); iter != clusters.end(); ++iter){
        for (const Cluster& cluster : iter->second){
            cluster.unique_indices(unique_indx);
        }
    }
}