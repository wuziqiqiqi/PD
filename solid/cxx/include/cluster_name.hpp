#ifndef CFNAME_H
#define CFNAME_H
#include <string>

struct ParsedName {
    // Container of a parsed name.
    unsigned int size;
    std::string prefix;
    std::string dec_str;
    // Dec num is only used for singlets.
    unsigned int dec_num = 0;
};

class ClusterName {
    /* Parser for Cluster names(or CF and ECI names, as
    these names are the same), e.g. c2_d0000_0_1 */
   public:
    ClusterName(const std::string &cf_name);

    /* Pre-parse the name into a single container object. */
    ParsedName get_parsed();

    /* Return the full name */
    std::string &get_name();

    /* Get the decoration number, for 1-body clusters, since this is
    always just one number. Does not verify this is actually a singlet.
    Example: c1_1 returns 1. */
    unsigned int get_dec_num();

    /* Get the corresponding cluster size
    Example: c2_d0000_0_1 returns 2 */
    unsigned int get_size();

    /* Extract the prefix and the decoration string from the name, as strings.
    Example:
    c2_d0000_1 becomes prefix = "c2_d0000_0" and dec_str = "1" */
    void get_prefix_and_dec_str(std::string &prefix, std::string &dec_str);

    /* Extract the prefix
    Example:
    c2_d0000_0_1 returns "c2_d0000_0" */
    std::string get_prefix();

    /* Get the decoration string
    Example:
    c3_d0000_0_010 returns "010" */
    std::string get_dec_str();

   private:
    // Reference to the name of the cluster
    std::string m_cf_name;
    // Position of the last separator
    int last_sep_pos();
    // Position of the first separator
    int first_sep_pos();
};

#endif
