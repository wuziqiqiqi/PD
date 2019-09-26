from setuptools import setup, find_packages, Extension
from distutils.errors import CompileError
from distutils import ccompiler
from textwrap import dedent
from distutils.sysconfig import get_python_inc
from pip._internal import main as pipmain
import os

def src_folder():
    candidates = ['cxx/src/', 'clease/', './']
    for c in candidates:
        if os.path.exists(c+'cluster.cpp'):
            return c
    raise RuntimeError("Cannot find source folder.")

def get_cython_folder():
    candidates = ['cxx/cython/', 'cython/', './']
    for c in candidates:
        if os.path.exists(c+'clease_cxx.pyx'):
            return c
    raise RuntimeError("Cannot find cython folder")

def include_folder():
    candidates = ['cxx/include/', 'clease/', './']
    for c in candidates:
        if os.path.exists(c + 'cluster.hpp'):
            return c
    raise RuntimeError("Cannot find include folder")

def get_npy_include_folder():
    try:
        import numpy
    except ImportError:
        pipmain(['install', 'numpy'])

    import numpy as np
    return np.get_include()

def build_ext(ext_module):
    try:
        import Cython
    except ImportError:
        pipmain(['install', 'cython'])

    from Cython.Build import cythonize

    return cythonize(ext_module)

def install_kivy_garden_from_github():
    try:
        from kivy_garden import Graph
    except ImportError:
        pipmain(['install', 'https://github.com/kivy-garden/graph/archive/master.zip'])


cxx_src_folder = src_folder()
cxx_inc_folder = include_folder()
cython_folder = get_cython_folder()
install_kivy_garden_from_github()

src_files = ['cf_history_tracker.cpp',
             'additional_tools.cpp', 'cluster.cpp',
             'row_sparse_struct_matrix.cpp',
             'named_array.cpp', 'symbols_with_numbers.cpp',
             'basis_function.cpp', 'cluster_list.cpp']

src_files = [cxx_src_folder + x for x in src_files]
src_files.append(cython_folder + 'clease_cxx.pyx')
extra_comp_args = ['-std=c++11']


def find_layout_files():
    folder = 'clease/gui/layout/'
    files = os.listdir(folder)
    return [folder + x for x in files if x.endswith('.kv')]


clease_cxx = Extension("clease_cxx", sources=src_files,
                       include_dirs=[cxx_inc_folder, get_npy_include_folder(),
                                     cxx_src_folder, get_python_inc(),
                                     cython_folder],
                       extra_compile_args=extra_comp_args,
                       language="c++")


setup(
    name="clease",
    setup_requires=['cython', 'numpy'],
    ext_modules=build_ext(clease_cxx),
    author=['J. H. Chang', 'D. Kleiven'],
    author_email="jchang@dtu.dk, david.kleiven@ntnu.no",
    long_description='CLuster Expansion in Atomistic Simulation Environment',
    url='https://gitlab.com/computationalmaterials/clease',
    scripts=['bin/clease'],
    version='0.9.2',
    description="CLuster Expansion in Atomistic Simulation Environment",
    packages=find_packages(),
    download_url='https://gitlab.com/computationalmaterials/clease/-/archive/v0.9.2/clease-v0.9.2.zip',
    include_package_data=True,
    package_data={'clease.gui': ['layout/*.kv']},
    license='MPL-2.0',
    keywords=['Cluster Expansion', 'Monte Carlo', 'Computational materials', 'Materials research'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=['ase', 'matplotlib', 'spglib', 'kivy']
)
