from setuptools import setup, find_packages, Extension
import numpy as np
from Cython.Build import cythonize
from distutils.errors import CompileError, LinkError
from distutils.compiler import ccompiler
from textwrap import dedent

cxx_src_folder = 'cxx/src/'
cxx_inc_folder = 'cxx/include'

src_files = ['cf_history_tracker.cpp',
             'additional_tools.cpp', 'cluster.cpp',
             'row_sparse_struct_matrix.cpp',
             'named_array.cpp', 'symbols_with_numbers.cpp',
             'basis_function.cpp']

src_files = [cxx_src_folder + x for x in src_files]
src_files.append('clease/cython/clease_cxx.pyx')
extra_comp_args = ['-std=c++11']


def check_python_development_headers():
    compiler = ccompiler.new_compiler()
    code = dedent(
        """
        #include <Python.h>

        int main(int argc, char **argv){
            return 0;
        };
        """
    )

    fname = "devel_header.cpp"
    with open(fname, 'w') as outfile:
        outfile.write(code)

    ok = True
    try:
        compiler.compile([fname])
    except CompileError:
        ok = False

    binfile = fname.split('.')[0] + '.o'
    try:
        os.remove(fname)
        os.remove(binfile)
    except Exception:
        pass
    return ok


if not check_python_development_headers():
    raise ValueError("Python development header needs to be available")


clease_cxx = Extension("clease_cxx", sources=src_files,
                       include_dirs=[cxx_inc_folder, np.get_include(),
                                     cxx_src_folder],
                       extra_compile_args=extra_comp_args,
                       language="c++")
setup(
    name="clease",
    ext_modules=cythonize(clease_cxx),
    version=1.0,
    description="Cluster Expansion In Atomistic Simulation Environment",
    packages=find_packages()
)
