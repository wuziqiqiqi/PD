#include "additional_tools.hpp"
#include "cf_history_tracker.hpp"
#include "symbol_change.hpp"
#include <stdexcept>
#include <sstream>

using namespace std;

ostream &operator<<(ostream &out, const SymbolChange &symb)
{
  out << "(Index: " << symb.indx << " old symbol: " << symb.old_symb << " new symbol: " << symb.new_symb << " track index: " << symb.track_indx << ")";
  return out;
}

std::ostream &operator<<(ostream &out, const array<SymbolChange, 2> &move)
{
  out << move[0] << "->" << move[1];
  return out;
}

int kronecker(int i, int j)
{
  return i == j ? 1 : 0;
};

PyObject *string2py(const string &str)
{
#if PY_MAJOR_VERSION >= 3
  // Python 3
  return PyUnicode_FromString(str.c_str());
#else
  // Python 2
  return PyUnicode_FromString(str.c_str());
#endif
}

string py2string(PyObject *str)
{
#if PY_MAJOR_VERSION >= 3
  // Python 3
  const char *char_str;
  if (PyUnicode_Check(str))
  {
    char_str = PyUnicode_AsUTF8(str);
  }
  else if (PyBytes_Check(str))
  {
    char_str = PyBytes_AsString(str);
  }
  else
  {
    throw invalid_argument("Unrecognized string type!");
  }

  if (char_str == NULL)
  {
    throw invalid_argument("Could not convert python string to const char");
  }
  return char_str;
#else
  // Python 2
  return PyString_AsString(str);
#endif
}

PyObject *int2py(int integer)
{
#if PY_MAJOR_VERSION >= 3
  return PyLong_FromLong(integer);
#else
  return PyInt_FromLong(integer);
#endif
}

int py2int(PyObject *integer)
{
#if PY_MAJOR_VERSION >= 3
  return PyLong_AsLong(integer);
#else
  return PyInt_AsLong(integer);
#endif
}

int int_attr_from_py_object(PyObject *obj, const char *attr_name)
{
  // Get an integer with attr_name. Ensure we dereference the reference again
  PyObject *ref = PyObject_GetAttrString(obj, attr_name);
  int value = py2int(ref);
  Py_DECREF(ref);
  return value;
}

std::string str_attr_from_py_object(PyObject *obj, const char *attr_name)
{
  // Get a string with attr_name. Ensure we dereference the reference again
  PyObject *ref = PyObject_GetAttrString(obj, attr_name);
  std::string value = py2string(ref);
  Py_DECREF(ref);
  return value;
}

void py_changes2symb_changes(PyObject *all_changes, vector<SymbolChange> &symb_changes)
{
  SymbolChange symb_change;
  unsigned int size = list_size(all_changes);
  for (unsigned int i = 0; i < size; i++)
  {
    symb_change = SymbolChange(PyList_GetItem(all_changes, i));
    symb_changes.push_back(symb_change);
  }
}

void py_change2swap_move(PyObject *all_changes, swap_move &symb_changes)
{
  SymbolChange symb_change;
  unsigned int size = list_size(all_changes);
  for (unsigned int i = 0; i < size; i++)
  {
    symb_changes[i] = SymbolChange(PyList_GetItem(all_changes, i));
  }
}

PyObject *get_attr(PyObject *obj, const char *name)
{
  PyObject *attr = PyObject_GetAttrString(obj, name);
  if (attr == nullptr)
  {
    stringstream ss;
    ss << "Python object has not attribute " << name;
    throw invalid_argument(ss.str());
  }
  return attr;
}

unsigned int list_size(PyObject *list)
{
  if (!PyList_Check(list))
  {
    throw invalid_argument("Python object is not a list. Cannot retrieve the length!");
  }
  return PyList_Size(list);
}

void inverse3x3(const mat3x3 &mat, mat3x3 &inv)
{
  // Set the inverse matrix equal to the identity matrix
  double determinant = 0;

  // finding determinant
  for (unsigned int i = 0; i < 3; i++)
    determinant = determinant + (mat[0][i] * (mat[1][(i + 1) % 3] * mat[2][(i + 2) % 3] - mat[1][(i + 2) % 3] * mat[2][(i + 1) % 3]));

  for (unsigned int i = 0; i < 3; i++)
  {
    for (unsigned int j = 0; j < 3; j++)
      inv[i][j] = ((mat[(j + 1) % 3][(i + 1) % 3] * mat[(j + 2) % 3][(i + 2) % 3]) - (mat[(j + 1) % 3][(i + 2) % 3] * mat[(j + 2) % 3][(i + 1) % 3])) / determinant;
  }
}
