#ifndef BASIS_FUNCTION_H
#define BASIS_FUNCTION_H
#include <iostream>
#include <map>
#include <vector>

#include "symbols_with_numbers.hpp"

typedef std::map<std::string, double> dict_dbl_t;
typedef std::vector<dict_dbl_t> bf_raw_t;

struct BFChange {
    BFChange(double new_bf, double old_bf) : new_bf(new_bf), old_bf(old_bf){};
    double new_bf;
    double old_bf;
};
class BasisFunction {
   public:
    BasisFunction();
    BasisFunction(const bf_raw_t &raw_bfs, const Symbols &symb_with_num);
    BasisFunction(const BasisFunction &other);
    BasisFunction &operator=(const BasisFunction &other);
    ~BasisFunction();

    /* Return the basis function value for a given decoration number and symbol ID */
    inline double get(unsigned int dec_num, unsigned int symb_id) const {
        /* This access is used in the inner loop of the spin product calculation (i.e. very
        frequently), so we access with no bounds checking for performance reasons.
        Also inline this method, as it's called very frequently in the hot path.
        */
        return bfs[get_index(dec_num, symb_id)];
    };

    /* Return the size (number of basis functions) */
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
    std::vector<BFChange> prepare_bfs_new_old(unsigned int new_id, unsigned int old_id) const;

    /** Stream operator */
    friend std::ostream &operator<<(std::ostream &out, const BasisFunction &bf);

   private:
    const Symbols *symb_ptr{nullptr};
    std::vector<double> bfs;
    unsigned int num_bfs{0};
    unsigned int num_bf_values{0};

    /** Return the corresponding index into the flattened array */
    inline unsigned int get_index(unsigned int dec_num, unsigned int symb_id) const {
        // Inline this method, as it's called very frequently in the hot path.
        return dec_num * this->num_bf_values + symb_id;
    }
    void swap(const BasisFunction &other);
};

#endif
