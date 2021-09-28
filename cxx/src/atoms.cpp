#include "config.hpp"
#include <vector>
#include "atoms.hpp"
#include "four_vector.hpp"
#include "additional_tools.hpp"
#include <Python.h>

using namespace std;

Atoms::Atoms(PyObject *py_atoms, PyObject *py_four_vectors)
{
    // Atoms object represented in the 4-vector space
    // Parse objects from Python format to native
    this->atoms = py_atoms;
#ifdef PRINT_DEBUG
    cout << "Found " << this->num_atoms() << " atoms." << endl;
#endif
    parse_four_vectors(py_four_vectors);
    parse_max_lattice();
}

void Atoms::parse_four_vectors(PyObject *py_list)
{

#ifdef PRINT_DEBUG
    cout << "Parsing four-vectors in Atoms" << endl;
#endif
    // (N, 4) object list, where N is the number of atoms
    // PySequence_Fast (new reference)
    PyObject *seq = PySequence_Fast(py_list, "Four-vectors must be iterable.");

    int N = PySequence_Fast_GET_SIZE(seq);
#ifdef PRINT_DEBUG
    cout << "Found " << N << " four-vectors." << endl;
#endif
    four_vectors.clear();
    four_vectors.reserve(N);

    for (int i = 0; i < N; i++)
    {
        // Get each python FourVector object (borrowed reference)
        PyObject *py_four_vec = PySequence_Fast_GET_ITEM(seq, i);
        // Parse the 4-vector
        int ix, iy, iz, sublattice;

        ix = int_attr_from_py_object(py_four_vec, "ix");
        iy = int_attr_from_py_object(py_four_vec, "iy");
        iz = int_attr_from_py_object(py_four_vec, "iz");
        sublattice = int_attr_from_py_object(py_four_vec, "sublattice");
        four_vectors.push_back({ix, iy, iz, sublattice});
    }
    Py_DECREF(seq);
}

// Find Nx, Ny, Nz and Ns
// i.e. the maximum repition in the x, y, and z directions,
// as well as the number of sublattices (number of sites in the primitive)
void Atoms::parse_max_lattice()
{
    Ns = 0, Nx = 0, Ny = 0, Nz = 0;
    for (auto fv : four_vectors)
    {
        Nx = (fv.ix > Nx) ? fv.ix : Nx;
        Ny = (fv.iy > Ny) ? fv.iy : Ny;
        Nz = (fv.iz > Nz) ? fv.iz : Nz;
        Ns = (fv.sublattice > Ns) ? fv.sublattice : Ns;
    }
    // Number of sublattices/repitiions is 1 greater than the max index,
    // since we start from 0.
    Ns++;
    Nx++;
    Ny++;
    Nz++;
}

unsigned int Atoms::num_atoms() const
{
    int n = PySequence_Length(this->atoms);
    return n;
}

PyObject *Atoms::get_atom(const int index) const
{
    return PySequence_GetItem(this->atoms, index);
}

std::vector<int> Atoms::get_numbers() const
{
    int num_atoms = this->num_atoms();

    std::vector<int> numbers;
    numbers.reserve(num_atoms);

    for (int i = 0; i < num_atoms; i++)
    {
        // Get atom number i (new reference).
        PyObject *atom = this->get_atom(i);
        PyObject *number = get_attr(atom, "number");
        int value = py2int(number);
        numbers.push_back(value);
        Py_DECREF(atom);
        Py_DECREF(number);
    }
    return numbers;
}

vector<string> Atoms::get_symbols() const
// Get the atomic numbers of the internal atoms object
{
    int num_atoms = this->num_atoms();

    std::vector<std::string> symbols;
    symbols.reserve(num_atoms);

    for (int i = 0; i < num_atoms; i++)
    {
        // Get atom number i (new reference).
        PyObject *atom = this->get_atom(i);
        PyObject *sym = get_attr(atom, "symbol");
        string value = py2string(sym);
        symbols.push_back(value);
        Py_DECREF(atom);
        Py_DECREF(sym);
    }
    return symbols;
}

const std::vector<FourVector> &Atoms::get_four_vectors() const
{
    return this->four_vectors;
}

int Atoms::get_1d_index(const FourVector &v) const
{
    // Convert a 4-vector into its 1d array index
    return v.ix * Ny * Nz * Ns + v.iy * Nz * Ns + v.iz * Ns + v.sublattice;
}

void Atoms::apply_change(PyObject *single_change)
{
    SymbolChange change = py_tuple_to_symbol_change(single_change);
    this->apply_change(change);
}

void Atoms::apply_change(const SymbolChange &single_change)
{
    this->set_symbol(single_change.new_symb, single_change.indx);
}

void Atoms::undo_change(PyObject *single_change)
{
    SymbolChange change = py_tuple_to_symbol_change(single_change);
    this->undo_change(change);
}

void Atoms::undo_change(const SymbolChange &single_change)
{
    this->set_symbol(single_change.old_symb, single_change.indx);
}

void Atoms::set_symbol(const string symb_str, const int index)
{
    PyObject *py_str = string2py(symb_str.c_str());
    PyObject *atom = this->get_atom(index);
    PyObject_SetAttrString(atom, "symbol", py_str);
    Py_DECREF(py_str);
    Py_DECREF(atom);
}
