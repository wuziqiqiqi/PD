#ifndef BASIS_FUNCTION_H
#define BASIS_FUNCTION_H
#include <iostream>
#include <map>
#include <vector>

#include "symbols_with_numbers.hpp"

typedef std::map<std::string, double> dict_dbl_t;
typedef std::vector<dict_dbl_t> bf_raw_t;

class BasisFunction {
   public:
    BasisFunction();
    BasisFunction(const bf_raw_t &raw_bfs, const Symbols &symb_with_num);
    BasisFunction(const BasisFunction &other);
    BasisFunction &operator=(const BasisFunction &other);
    ~BasisFunction();

    /** Return the basis function value for a given decoration number and symbol ID */
    double get(unsigned int dec_num, unsigned int symb_id) const;

    /** Return the size (number of basis functions) */
    unsigned int size() const {
        return num_bfs;
    };

    /* For preparing a vector of basis functions that maps a decoration number
    to the old and new basis functions simultaneously.
    This vector contains copies of the underlying basis functions.
    The pair<double, double> is constructed as follows:
    first -> Basis function of the *new* symbol
    second -> Basis function of the *old* symbol
    */
    std::vector<std::pair<double, double>> prepare_bfs_new_old(unsigned int new_id,
                                                               unsigned int old_id) const;

    /** Stream operator */
    friend std::ostream &operator<<(std::ostream &out, const BasisFunction &bf);

   private:
    const Symbols *symb_ptr{nullptr};
    std::vector<double> bfs;
    unsigned int num_bfs{0};
    unsigned int num_bf_values{0};

    /** Return the corresponding index into the flattened array */
    unsigned int get_index(unsigned int dec_num, unsigned int symb_id) const;
    void swap(const BasisFunction &other);
};

#endif
