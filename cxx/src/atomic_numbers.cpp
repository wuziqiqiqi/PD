#include "atomic_numbers.hpp"
#include "additional_tools.hpp"
#include <iostream>
#include <stdexcept>

// Lookup the number from a symbol - will throw a runtime_error exception if the symbol was not found,
// rather than just returning some random integer (often 0!)
int get_number_from_symbol(std::string symbol)
{
    auto it = atomic_numbers_map.find(symbol);
    if (it == atomic_numbers_map.end())
    {
        // Symbol not found, raise an exception, rather than quietly returning non-sense
        throw std::runtime_error("Atomic symbol '" + symbol + "' not found.");
    }
    return it->second; // The actual atomic number
}

// A fast way of adjusting a single symbol in a Python atoms object,
// by looking up the symbol number in a table in C++, and setting the new number directly.
// Roughly equivalent to (in Python)
// from ase.data import atomic numbers
// atoms.numbers[index] = atomic_numbers[symbol]
void set_symbol_in_atoms(PyObject *py_atoms, int index, std::string symbol)
{
    PyObject *numbers = PyObject_GetAttrString(py_atoms, "numbers"); // Get the numbers array, i.e. atoms.numbers
    int num = get_number_from_symbol(symbol);                        // Look up the corresponding atomic numbers
#ifdef PRINT_DEBUG
    std::cout << "Setting new atomic number at index " << index << " to " << num << std::endl;
#endif
    PyObject *new_number = int2py(num);                        // Convert the atomic number into a python object
    int code = PySequence_SetItem(numbers, index, new_number); // Insert the new number
    // some cleanup
    Py_DECREF(numbers);
    Py_DECREF(new_number);
    // Check the SetItem was successful. PySequence_SetItem returns 0 on success
    if (code != 0)
    {
        // Crash, notifying that something went wrong.
        throw std::runtime_error("Updating atomic symbol failed.");
    }
}
