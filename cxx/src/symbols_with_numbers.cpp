#include "symbols_with_numbers.hpp"
#include <cstring>
#include <iostream>

Symbols::Symbols(const vec_str_t &symbs, const set_str_t &unique_symbs) : symbols(symbs)
{
    symb_ids.resize(symbs.size());

#ifdef PRINT_DEBUG
    std::cout << "Symbols size " << symb_ids.size() << std::endl;
#endif

    unsigned int current_id = 0;
    for (auto iter = unique_symbs.begin(); iter != unique_symbs.end(); ++iter)
    {
        symb_id_translation[*iter] = current_id++;
    }

    // Populate the symb_id array
    update_ids();
}

Symbols::Symbols(const Symbols &other)
{
    other.swap(*this);
}

Symbols &Symbols::operator=(const Symbols &other)
{
    other.swap(*this);
    return *this;
}

bool Symbols::is_consistent() const
{
    for (unsigned int i = 0; i < symbols.size(); i++)
    {
        if (symb_ids.at(i) != symb_id_translation.at(symbols[i]))
        {
            return false;
        }
    }
    return true;
}

void Symbols::set_symbol(unsigned int indx, const std::string &new_symb)
{
    symbols[indx] = new_symb;
    symb_ids[indx] = symb_id_translation[new_symb];
}

void Symbols::set_symbols(const vec_str_t &new_symbs)
{
    symb_ids.resize(new_symbs.size());
    symbols = new_symbs;
    update_ids();
}

void Symbols::update_ids()
{
    for (unsigned int i = 0; i < symbols.size(); i++)
    {
        symb_ids.at(i) = symb_id_translation[symbols[i]];
    }
}

void Symbols::swap(Symbols &other) const
{
    other.symbols = symbols;
    other.symb_id_translation = symb_id_translation;
    other.symb_ids = symb_ids;
}
