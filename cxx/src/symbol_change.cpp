#include "symbol_change.hpp"
#include "additional_tools.hpp"

SymbolChange::SymbolChange(int indx, const std::string &old_symb, const std::string &new_symb)
    : indx(indx), old_symb(old_symb), new_symb(new_symb) {}

/* Transfer a python SystemChange object to the C++ equivalent */
SymbolChange::SymbolChange(PyObject *py_change)
{
    // Transfer data into this class
    this->indx = int_attr_from_py_object(py_change, "index");
    this->old_symb = str_attr_from_py_object(py_change, "old_symb");
    this->new_symb = str_attr_from_py_object(py_change, "new_symb");
}
