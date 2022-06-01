#include "atomic_numbers.hpp"

#include <iostream>

#include "additional_tools.hpp"

// A fast way of adjusting a single symbol in a Python atoms object,
// by looking up the symbol number in a table in C++, and setting the new number directly.
// Roughly equivalent to (in Python)
// from ase.data import atomic numbers
// atoms.numbers[index] = atomic_numbers[symbol]
void set_symbol_in_atoms(PyObject *py_atoms, const int index, const std::string &symbol) {
    // Get the numbers array, i.e. atoms.numbers
    PyObject *numbers = PyObject_GetAttrString(py_atoms, "numbers");
    // Lookup the number from a symbol, which will throw an std::out_of_range error
    // if the symbol doesn't exist.
    int num = atomic_numbers_map.at(symbol);  // Look up the corresponding atomic numbers
#ifdef PRINT_DEBUG
    std::cout << "Setting new atomic number at index " << index << " to " << num << std::endl;
#endif
    PyObject *new_number = int2py(num);  // Convert the atomic number into a python object
    int code = PySequence_SetItem(numbers, index, new_number);  // Insert the new number
    // some cleanup
    Py_DECREF(numbers);
    Py_DECREF(new_number);
    // Check the SetItem was successful. PySequence_SetItem returns 0 on success
    if (code != 0) {
        // Crash, notifying that something went wrong.
        throw std::runtime_error("Updating atomic symbol failed.");
    }
}

std::unordered_map<int, std::string> get_number_to_symbol_map() {
    std::unordered_map<int, std::string> reversed_map;
    for (const auto &it : atomic_numbers_map) {
        reversed_map[it.second] = it.first;
    }
    return reversed_map;
};

std::vector<std::string> get_symbols_from_atoms(PyObject *py_atoms) {
    // Read the symbols by translating the numbers array to symbols.
    PyObject *py_numbers = PyObject_GetAttrString(py_atoms, "numbers");
    unsigned int n_atoms = PyObject_Length(py_numbers);

    auto num_to_sym = get_number_to_symbol_map();

    std::vector<std::string> symbols;
    symbols.reserve(n_atoms);
    // Translate the numbers array into symbols
    for (unsigned int i = 0; i < n_atoms; i++) {
        PyObject *pynum = PySequence_GetItem(py_numbers, i);
        if (pynum == NULL) {
            throw std::runtime_error("Could not retrieve element " + std::to_string(i));
        }
        int atom_no = py2int(pynum);

        symbols.emplace_back(num_to_sym.at(atom_no));
        Py_DECREF(pynum);
    }
    Py_DECREF(py_numbers);

#ifdef PRINT_DEBUG
    std::cout << "Symbols read from atoms:\n";
    for (auto &s : symbols) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
#endif
    return symbols;
}
