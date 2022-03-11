#ifndef SYMBOL_CHANGE_H
#define SYMBOL_CHANGE_H

#include <Python.h>
#include <string>

class SymbolChange
{
public:
    int indx;
    std::string old_symb;
    std::string new_symb;
    int track_indx{0};

    SymbolChange() : indx(0), old_symb(""), new_symb(""){};
    SymbolChange(int indx, const std::string &old_symb, const std::string &new_symb);
    SymbolChange(PyObject *py_change);
};

#endif
