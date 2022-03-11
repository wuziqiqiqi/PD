#include "config.hpp"
#include <vector>
#include "atoms.hpp"
#include "four_vector.hpp"
#include "additional_tools.hpp"
#include <Python.h>

using namespace std;

Atoms::Atoms(PyObject *py_atoms, PyObject *py_four_vectors)
    : atoms(py_atoms)
{
    // Atoms object represented in the 4-vector space
    // Parse objects from Python format to native
#ifdef PRINT_DEBUG
    cout << "Found " << this->num_atoms() << " atoms." << endl;
#endif
    parse_four_vectors(py_four_vectors);
}

void Atoms::parse_four_vectors(PyObject *py_four_vectors)
{

#ifdef PRINT_DEBUG
    cout << "Parsing four-vectors in Atoms" << endl;
#endif
    // Sequence of length N, where N is the number of atoms
    // containing FourVector Objects
    // PySequence_Fast (new reference)
    PyObject *seq = PySequence_Fast(py_four_vectors, "Four-vectors must be iterable.");

    Py_ssize_t N = PySequence_Fast_GET_SIZE(seq);
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
        int ix = int_attr_from_py_object(py_four_vec, "ix");
        int iy = int_attr_from_py_object(py_four_vec, "iy");
        int iz = int_attr_from_py_object(py_four_vec, "iz");
        int sublattice = int_attr_from_py_object(py_four_vec, "sublattice");

        // Use emplace_back to avoid a copy. Create the four-vector directly
        // inside the vector object.
        four_vectors.emplace_back(ix, iy, iz, sublattice);
    }
    Py_DECREF(seq);
    // Update the lattice information.
    parse_max_lattice();
}

// Find Nx, Ny, Nz and Ns
// i.e. the maximum repetition in the x, y, and z directions,
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
    // Number of sublattices/repetitions is 1 greater than the max index,
    // since we start from 0.
    Ns++;
    Nx++;
    Ny++;
    Nz++;
}

Py_ssize_t Atoms::num_atoms() const
{
    return PySequence_Length(this->atoms);
}

PyObject *Atoms::get_atom(const Py_ssize_t index) const
{
    return PySequence_GetItem(this->atoms, index);
}

std::vector<int> Atoms::get_numbers() const
{
    Py_ssize_t num_atoms = this->num_atoms();

    std::vector<int> numbers;
    numbers.reserve(num_atoms);

    for (int i = 0; i < num_atoms; ++i)
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
    SymbolChange change = SymbolChange(single_change);
    this->apply_change(change);
}

void Atoms::apply_change(const SymbolChange &single_change)
{
    this->set_symbol(single_change.new_symb, single_change.indx);
}

void Atoms::undo_change(PyObject *single_change)
{
    SymbolChange change = SymbolChange(single_change);
    this->undo_change(change);
}

void Atoms::undo_change(const SymbolChange &single_change)
{
    this->set_symbol(single_change.old_symb, single_change.indx);
}

void Atoms::set_symbol(const string &symb_str, const Py_ssize_t index)
{
    PyObject *py_str = string2py(symb_str.c_str());
    PyObject *atom = this->get_atom(index);
    PyObject_SetAttrString(atom, "symbol", py_str);
    Py_DECREF(py_str);
    Py_DECREF(atom);
}
