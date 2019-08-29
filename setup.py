from setuptools import setup, find_packages, Extension
import numpy as np
from Cython.Build import cythonize
from distutils.errors import CompileError
from distutils import ccompiler
from textwrap import dedent
from distutils.sysconfig import get_python_inc
import os

cxx_src_folder = 'cxx/src/'
cxx_inc_folder = 'cxx/include'

src_files = ['cf_history_tracker.cpp',
             'additional_tools.cpp', 'cluster.cpp',
             'row_sparse_struct_matrix.cpp',
             'named_array.cpp', 'symbols_with_numbers.cpp',
             'basis_function.cpp', 'cluster_list.cpp']

src_files = [cxx_src_folder + x for x in src_files]
src_files.append('clease/cython/clease_cxx.pyx')
extra_comp_args = ['-std=c++11']


def check_python_development_headers():
    """
    Try to compile a small snippet in order to check if the Python development
    files are available
    """
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
        compiler.compile([fname], include_dirs=[get_python_inc()])
    except CompileError as exc:
        print(str(exc))
        ok = False

    binfile = fname.split('.')[0] + '.o'
    try:
        os.remove(fname)
        os.remove(binfile)
    except Exception:
        pass
    return ok


def find_layout_files():
    folder = 'clease/gui/layout/'
    files = os.listdir(folder)
    return [folder + x for x in files if x.endswith('.kv')]

if not check_python_development_headers():
    raise ValueError("Python development header must be available.")


clease_cxx = Extension("clease_cxx", sources=src_files,
                       include_dirs=[cxx_inc_folder, np.get_include(),
                                     cxx_src_folder, get_python_inc()],
                       extra_compile_args=extra_comp_args,
                       language="c++")


setup(
    name="clease",
    ext_modules=cythonize(clease_cxx),
    scripts=['bin/clease'],
    version='0.9.0',
    description="CLuster Expansion in Atomistic Simulation Environment",
    packages=find_packages(),
    download_url='https://gitlab.com/computationalmaterials/clease/-/archive/v0.9.0/clease-v0.9.0.zip',
    include_package_data=True,
    data_files=[('layout', find_layout_files())],
    license='MPL-2.0',
    keywords=['Cluster Expansion', 'Monte Carlo', 'Computational materials', 'Materials research'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',  
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)
