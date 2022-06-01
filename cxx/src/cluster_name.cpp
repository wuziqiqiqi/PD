#include "cluster_name.hpp"

#include <string>
ClusterName::ClusterName(const std::string &cf_name) {
    this->m_cf_name = cf_name;
}

ParsedName ClusterName::get_parsed() {
    unsigned int size = get_size();
    std::string prefix, dec_str;
    get_prefix_and_dec_str(prefix, dec_str);
    ParsedName parsed;
    parsed.size = size;
    parsed.prefix = prefix;
    parsed.dec_str = dec_str;

    // The decoration number is only well-defined for singlets.
    if (size == 1) {
        parsed.dec_num = std::stoi(dec_str);
    }
    return parsed;
}

std::string &ClusterName::get_name() {
    return this->m_cf_name;
}

unsigned int ClusterName::get_dec_num() {
    std::string dec_str = this->get_dec_str();
    // stoi: string-to-integer
    return std::stoi(dec_str);
}

unsigned int ClusterName::get_size() {
    // Example string: c2_d0000_0_1 should return 2
    int pos = this->first_sep_pos();
    // Start at pos 1 to skip the first "c"
    std::string size_str = this->m_cf_name.substr(1, pos);
    return std::stoi(size_str);
}

void ClusterName::get_prefix_and_dec_str(std::string &prefix, std::string &dec_str) {
    int pos = this->last_sep_pos();
    prefix = this->m_cf_name.substr(0, pos);
    dec_str = this->m_cf_name.substr(pos + 1);
};

std::string ClusterName::get_prefix() {
    int pos = this->last_sep_pos();
    return this->m_cf_name.substr(0, pos);
}

std::string ClusterName::get_dec_str() {
    int pos = this->last_sep_pos();
    return this->m_cf_name.substr(pos + 1);
}

int ClusterName::first_sep_pos() {
    return this->m_cf_name.find("_");
}

int ClusterName::last_sep_pos() {
    return this->m_cf_name.rfind("_");
}
