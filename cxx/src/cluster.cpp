#include "config.hpp"
#include "cluster.hpp"
#include "additional_tools.hpp"
#include <Python.h>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <set>

using namespace std;

Cluster::Cluster(PyObject *info_dict)
{
  parse_info_dict(info_dict);
}

ostream &operator<<(ostream &out, const Cluster &cluster)
{
  out << "Name: " << cluster.name << "\n";
  // out << "Descriptor: " << cluster.descriptor << "\n";
  out << "Max cluster dia: " << cluster.max_cluster_dia << "\n";
  out << "Size: " << cluster.size << "\n";
  out << "ref_indx: " << cluster.ref_indx << "\n";
  out << "Members:\n";
  out << cluster.get();
  out << "\nEquivalent sites:\n";
  out << cluster.get_equiv();
  return out;
}

void Cluster::deco2string(const vector<int> &deco, string &name)
{
  stringstream ss;
  for (unsigned int i = 0; i < deco.size(); i++)
  {
    ss << deco[i];
  }
  name = ss.str();
}

vector<vector<int>> Cluster::get_all_decoration_numbers(int n_basis_funcs) const
{
  vector<vector<int>> bf_indx;
  // Insert all decoration numbers into bf_indx
  all_deco(n_basis_funcs, bf_indx);
  return bf_indx;
}

void Cluster::construct_equivalent_deco(int n_basis_funcs)
{
#ifdef PRINT_DEBUG
  cout << "Reading equivalent deco...\n";
#endif
  vector<vector<int>> bf_indx = get_all_decoration_numbers(n_basis_funcs);
  if (equiv_sites.size() == 0)
  {
    // There are no equivalent sites, or there are only one basis function
    // Fill the lookup with empty vectors
    string deco_str;
    for (vector<int> &deco : bf_indx)
    {
      deco2string(deco, deco_str);
      vector<vector<int>> one_vector = {deco};
      equiv_deco[deco_str] = one_vector;
    }

#ifdef PRINT_DEBUG
    cout << "Finished reading equiv deco (no equiv sites)\n";
#endif
    return;
  }

  // Convert the equivalent sites to a list of lists
  PyObject *py_eq_sites = PyList_New(equiv_sites.size());
  for (unsigned int i = 0; i < equiv_sites.size(); i++)
  {
    if (equiv_sites[i].size() == 0)
    {
      throw runtime_error("One of the entries in equiv_sites are zero!");
    }

    PyObject *py_group = PyList_New(equiv_sites[i].size());
    for (unsigned int j = 0; j < equiv_sites[i].size(); j++)
    {
      PyObject *py_site = int2py(equiv_sites[i][j]);
      PyList_SetItem(py_group, j, py_site);
    }
    PyList_SetItem(py_eq_sites, i, py_group);
  }

  // Use the python methods due to the convenient itertools module
  string mod_name("clease.tools");
  PyObject *mod_string = string2py(mod_name);
  PyObject *ce_tools_mod = PyImport_Import(mod_string);
  PyObject *equiv_deco_func = PyObject_GetAttrString(ce_tools_mod, "equivalent_deco");

  if (equiv_deco_func == nullptr)
  {
    throw runtime_error("Could not import equivalanet_deco function!");
  }

  for (vector<int> &deco : bf_indx)
  {
    string deco_str;
    deco2string(deco, deco_str);

    // Convert the vector of int into a python list
    PyObject *dec_list = PyList_New(deco.size());
    for (unsigned int i = 0; i < deco.size(); i++)
    {
      PyObject *py_int = int2py(deco[i]);
      PyList_SetItem(dec_list, i, py_int);
    }

    PyObject *args = PyTuple_Pack(2, dec_list, py_eq_sites);
    PyObject *py_equiv = PyObject_CallObject(equiv_deco_func, args);

    // Create a nested vector based on the result from Python
    vector<vector<int>> eq_dec;
    unsigned int size = list_size(py_equiv);
    for (unsigned int i = 0; i < size; i++)
    {
      vector<int> one_dec;
      PyObject *py_one_dec = PyList_GetItem(py_equiv, i);
      for (unsigned int j = 0; j < deco.size(); j++)
      {
        one_dec.push_back(py2int(PyList_GetItem(py_one_dec, j)));
      }
      eq_dec.push_back(one_dec);
    }

    equiv_deco[deco_str] = eq_dec;
    Py_DECREF(py_equiv);
    Py_DECREF(args);
    Py_DECREF(dec_list);
  }

  Py_DECREF(mod_string);
  Py_DECREF(ce_tools_mod);
  Py_DECREF(equiv_deco_func);
  Py_DECREF(py_eq_sites);

#ifdef PRINT_DEBUG
  cout << "Finished reading equiv deco\n";
#endif
}

void Cluster::all_deco(int num_bfs, vector<vector<int>> &deco) const
/*
Construct all decoration numbers, i.e. all permutations of
[0, 1, ..., num_bfs] of length L, where L is the size
of the cluster
The decoration numbers are pushed back into the "deco" vector.
Number of permutations which are added: num_bfs**size
Example: size=3, num_bfs=2, we add the following vectors:
  0 0 0
  1 0 0
  0 1 0
  1 1 0
  0 0 1
  1 0 1
  0 1 1
  1 1 1
*/
{
  int size = get_size(); // Size of the cluster

  if (size <= 1)
  {
    return;
  }

  int increment, tot;
  // Total number of permutations:
  unsigned int num_perm = static_cast<unsigned int>(pow(num_bfs, size));

  // Reserve the number of permutations, reduce copying
  deco.reserve(num_perm);

  // Initialize a running vector of correct size, starting at [0, 0, ...]
  // We will be incrementing this starting from the left 1 by 1
  vector<int> running(size, 0);
  deco.push_back(running); // Push back the first 0's vector
  // Iterate 1 less, because we already did the first [0, 0, 0,  ...] vector
  for (unsigned int i = 0; i < (num_perm - 1); i++)
  {
    // Increment the running vector by 1 from the left,
    // and propagate any modulo "num_bfs" to the right
    increment = 1;

    for (int &x : running)
    {
      // Add the increment to the value,
      // And figure out how much we increment the next value
      tot = x + increment;
      x = tot % num_bfs;         // Update the current value
      increment = tot / num_bfs; // How much should be added to the value to the right?

      if (increment == 0)
      {
        // We don't need to continue if we're not adding anything to the
        // values to the right
        break;
      }
    }
    // Add the permutation to the decoration array
    deco.push_back(running);
  }
}

const equiv_deco_t &Cluster::get_equiv_deco(const string &dec_str) const
{
  return equiv_deco.at(dec_str);
}

const equiv_deco_t &Cluster::get_equiv_deco(const std::vector<int> &deco) const
{
  string dec_str;
  deco2string(deco, dec_str);
  return get_equiv_deco(dec_str);
}

void Cluster::parse_info_dict(PyObject *info)
{
#ifdef PRINT_DEBUG
  cout << "Reading ref_indx\n";
#endif
  // Read reference index
  PyObject *py_ref_indx = get_attr(info, "ref_indx");
  ref_indx = py2int(py_ref_indx);
  Py_DECREF(py_ref_indx);

#ifdef PRINT_DEBUG
  cout << "Reading size\n";
#endif
  // Read size
  PyObject *py_size = get_attr(info, "size");
  size = py2int(py_size);
  Py_DECREF(py_size);

#ifdef PRINT_DEBUG
  cout << "Reading max_cluster_dia\n";
#endif
  // Read max_cluster_dia
  PyObject *py_mx_dia = get_attr(info, "diameter");
  if (size <= 1)
  {
    max_cluster_dia = 0.0;
  }
  else
  {
    max_cluster_dia = PyFloat_AS_DOUBLE(py_mx_dia);
  }
  Py_DECREF(py_mx_dia);

#ifdef PRINT_DEBUG
  cout << "Read symm_group\n";
#endif
  // Read symmetry group
  PyObject *py_symm = get_attr(info, "group");
  symm_group = py2int(py_symm);
  Py_DECREF(py_symm);

#ifdef PRINT_DEBUG
  cout << "Read name\n";
#endif
  // Read the name
  PyObject *py_name = get_attr(info, "name");
  name = py2string(py_name);
  Py_DECREF(py_name);

#ifdef PRINT_DEBUG
  cout << "Reading indices...\n";
#endif
  // Read indices
  PyObject *py_indx = get_attr(info, "indices");
  nested_list_to_cluster(py_indx, figures);
  Py_DECREF(py_indx);

  // Read equivalent sites
  PyObject *py_equiv_sites = get_attr(info, "equiv_sites");
  nested_list_to_cluster(py_equiv_sites, equiv_sites);
  Py_DECREF(py_equiv_sites);

#ifdef PRINT_DEBUG
  cout << "Reading normalization factors\n";
#endif

  // Read normalization factors
  PyObject *cluster_info_dict = get_attr(info, "info");
  PyObject *key = string2py("normalization_factor");
  if (PyDict_Contains(cluster_info_dict, key))
  {
#ifdef PRINT_DEBUG
    cout << "Loading duplication factors from python dictionary\n";
#endif
    PyObject *py_dup_factors = PyDict_GetItemString(cluster_info_dict, "normalization_factor");
    calculate_scaling_factors(py_dup_factors);
  }
  else
  {
#ifdef PRINT_DEBUG
    cout << "Filling duplication factors with 1\n";
#endif
    // Fill duplication factors with 1.0
    duplication_factors.clear();
    for (unsigned int i = 0; i < figures.size(); i++)
    {
      duplication_factors.push_back(1.0);
    }
  }
#ifdef PRINT_DEBUG
  cout << "Normalization factors (amount: " << duplication_factors.size() << "):\n";
  for (const auto &v : duplication_factors)
  {
    cout << v << " ";
  }
  cout << std::endl;
#endif
  Py_DECREF(key);

  // Sanity check
  check_consistency();
}

void Cluster::nested_list_to_cluster(PyObject *py_list, cluster_t &vec)
{
  int size = list_size(py_list);
  for (int i = 0; i < size; i++)
  {
    vector<int> one_cluster;
    PyObject *seq = PySequence_Fast(PyList_GetItem(py_list, i), NULL); // New reference
    int n_memb = PySequence_Fast_GET_SIZE(seq);
    for (int j = 0; j < n_memb; j++)
    {
      one_cluster.push_back(py2int(PySequence_Fast_GET_ITEM(seq, j)));
    }
    Py_DECREF(seq);
    vec.push_back(one_cluster);
  }
}

void Cluster::check_consistency() const
{
  if (duplication_factors.size() != figures.size())
  {
    throw runtime_error("A duplication factor for each member is required!");
  }
}

void Cluster::calculate_scaling_factors(PyObject *pylist)
{
  duplication_factors.clear();
  for (unsigned int i = 0; i < PyList_Size(pylist); i++)
  {
    double factor = PyFloat_AsDouble(PyList_GetItem(pylist, i));

    // Calculate unique sites in the sub cluster
    set<int> unique_values;
    unique_values.insert(ref_indx);
    for (int indx : figures[i])
    {
      unique_values.insert(indx);
    }

    double scale = static_cast<double>(unique_values.size()) / (figures[i].size());
    duplication_factors.push_back(factor * scale);
  }
}

unsigned int Cluster::max_index() const
{
  unsigned int max = 0;
  for (const auto &vec : figures)
  {
    for (auto &value : vec)
    {
      if (value > max)
      {
        max = value;
      }
    }
  }
  return max;
}

void Cluster::unique_indices(set<int> &indices) const
{
  for (auto &vec : figures)
  {
    for (auto &value : vec)
    {
      indices.insert(value);
    }
  }
}
