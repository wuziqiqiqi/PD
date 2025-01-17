#ifndef ADDITIONAL_TOOLS_H
#define ADDITIONAL_TOOLS_H
#include <Python.h>

#include <array>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "cf_history_tracker.hpp"

typedef std::array<SymbolChange, 2> swap_move;
typedef std::array<std::array<double, 3>, 3> mat3x3;

template <class key, class value>
std::ostream &operator<<(std::ostream &out, const std::map<key, value> &map);

std::ostream &operator<<(std::ostream &out, const SymbolChange &symb);

std::ostream &operator<<(std::ostream &out, const std::array<SymbolChange, 2> &move);

template <class T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec);

template <class T>
std::ostream &operator<<(std::ostream &out, const std::set<T> &set);

template <class T>
std::vector<T> &cyclic_permute(std::vector<T> &vec);

template <class T, unsigned int N>
std::ostream &operator<<(std::ostream &out, const std::array<T, N> &array);

template <class T>
void keys(std::map<std::string, T> &, std::vector<std::string> &keys);

template <class T>
void set2vector(const std::set<T> &s, std::vector<T> &v);

int kronecker(int i, int j);

PyObject *string2py(const std::string &string);
std::string py2string(PyObject *str);

PyObject *int2py(int integer);
int py2int(PyObject *integer);

// Extract an integer from a Python object by attribute name
int int_attr_from_py_object(PyObject *obj, const char *attr_name);
// Extract a string from a Python object by attribute name
std::string str_attr_from_py_object(PyObject *obj, const char *attr_name);

PyObject *get_attr(PyObject *obj, const char *name);

/** Return the length of a python list */
unsigned int list_size(PyObject *list);

void py_changes2symb_changes(PyObject *all_changes, std::vector<SymbolChange> &symb_changes);
void py_change2swap_move(PyObject *all_changes, swap_move &symb_changes);

/** Return true if element is in vector*/
template <class T>
bool is_in_vector(const T &value, const std::vector<T> &vec);

template <class T>
void insert_in_set(const std::vector<T> &vec, std::set<T> &unique);

/** Calculate the inverse of a 3x3 matrix */
void inverse3x3(const mat3x3 &inarray, mat3x3 &inv);

/* Check if a string starts with a given substring */
bool str_starts_with(const std::string &str, const std::string &sub_str);

bool has_parallel();

#include "additional_tools.tpp"
#endif
