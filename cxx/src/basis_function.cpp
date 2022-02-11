#include "basis_function.hpp"
#include "additional_tools.hpp"
#include <vector>

using namespace std;

BasisFunction::BasisFunction(){};

BasisFunction::BasisFunction(const bf_raw_t &raw_bf_data, const Symbols &symb_with_num) : symb_ptr(&symb_with_num)
{
    // We do not store the raw_bf_data, so we need to extract as much as we need now.
    num_bfs = raw_bf_data.size();
    num_bf_values = symb_ptr->num_unique_symbols();

    this->bfs.clear();
    this->bfs.resize(num_bfs * num_bf_values);

    // Transfer the raw bf array to the flattened array
    for (unsigned int dec_num = 0; dec_num < num_bfs; dec_num++)
        for (auto iter = raw_bf_data[dec_num].begin(); iter != raw_bf_data[dec_num].end(); ++iter)
        {
            unsigned int indx = get_index(dec_num, symb_ptr->get_symbol_id(iter->first));
            /* Assign with bounds check, since this isn't performance critical,
            as this construction is only done during initialization. */
            this->bfs.at(indx) = iter->second;
        }
};

BasisFunction::BasisFunction(const BasisFunction &other)
{
    this->swap(other);
}

BasisFunction &BasisFunction::operator=(const BasisFunction &other)
{
    this->swap(other);
    return *this;
}

BasisFunction::~BasisFunction()
{
}

unsigned int BasisFunction::get_index(unsigned int dec_num, unsigned int symb_id) const
{
    return dec_num * this->num_bf_values + symb_id;
}

double BasisFunction::get(unsigned int dec_num, unsigned int symb_id) const
{
    /* This access is used in the inner loop of the spin product calculation (i.e. very frequently),
     so we access with no bounds checking for performance reasons. */
    return bfs[get_index(dec_num, symb_id)];
}

void BasisFunction::swap(const BasisFunction &other)
{
    this->symb_ptr = other.symb_ptr;
    this->num_bfs = other.num_bfs;
    this->num_bf_values = other.num_bf_values;
    this->bfs = other.bfs;
}

ostream &operator<<(ostream &out, const BasisFunction &bf)
{
    out << "Basis Function object\n";
    out << "Number of elements: " << bf.size() << "\n";
    out << "Flattened array\n";
    for (unsigned int i = 0; i < bf.size(); i++)
    {
        out << bf.bfs[i] << " ";
    }
    return out;
}
