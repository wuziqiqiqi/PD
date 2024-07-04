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

    /** Stream operator */
    friend std::ostream &operator<<(std::ostream &out, const BasisFunction &bf);

   private:
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
