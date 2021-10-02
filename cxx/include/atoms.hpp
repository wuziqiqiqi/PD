#ifndef CLEASE_ATOMS_H
#define CLEASE_ATOMS_H

#include "four_vector.hpp"
#include "additional_tools.hpp"

#include <vector>
#include <Python.h>

class Atoms
{
public:
    int Ns, Nx, Ny, Nz;

    Atoms(PyObject *py_atoms, PyObject *four_vectors);

    // Get the atomic numbers of the internal atoms object
    std::vector<int> get_numbers() const;
    // Get the symbols of the internal atoms object
    std::vector<std::string> get_symbols() const;
    const std::vector<FourVector> &get_four_vectors() const;
    // Convert a 4-vector into its 1d array index
    int get_1d_index(const FourVector &v) const;
    Py_ssize_t num_atoms() const;

    void apply_change(const SymbolChange &single_change); // Mutate the internal atoms object, by applying a change
    void apply_change(PyObject *single_change);           // Apply SystemChange object from CLEASE

    void undo_change(const SymbolChange &single_change); // Mutate the internal atoms object, by undoing a change
    void undo_change(PyObject *single_change);           // Undo a SystemChange object from CLEASE

private:
    // Internal representation of the Atoms by its 4-vector positions and atomic numbers

    std::vector<FourVector> four_vectors;
    PyObject *atoms{nullptr}; // Pointer to the Python atoms object
    /* Get atom with index i.
    Remember: This creates a new reference,
    which must be Py_DECREF'ed when no longer in use.
    */
    PyObject *get_atom(const Py_ssize_t index) const;
    // Set the symbol of index i to a new symbol value.
    void set_symbol(const std::string &symb_str, const Py_ssize_t index);

    // Inititialization and updating
    void parse_four_vectors(PyObject *py_list);
    void parse_max_lattice();
};

#endif
