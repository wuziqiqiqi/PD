C++ files which rely on NumPy are placed in here, as they cannot be compiled directly.
They must be compiled after Cython has initialized the numpy arrays, which
happens in the `cxx/cython/clease_cxx.pyx` file.
Thus, any files which are to be included that rely on NumPy must be explicitly included in that
file as well.
