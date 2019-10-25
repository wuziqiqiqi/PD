import os
import re
from setuptools import setup, find_packages, Extension
from distutils.sysconfig import get_python_inc


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
    import numpy as np
    return np.get_include()


def build_ext(ext_module):
    from Cython.Build import cythonize
    return cythonize(ext_module)

# Get version number
with open('clease/__init__.py') as fd:
    version = re.search("__version__ = '(.*)'", fd.read()).group(1)


cxx_src_folder = src_folder()
cxx_inc_folder = include_folder()
cython_folder = get_cython_folder()

src_files = ['cf_history_tracker.cpp',
             'additional_tools.cpp', 'cluster.cpp',
             'row_sparse_struct_matrix.cpp',
             'named_array.cpp', 'symbols_with_numbers.cpp',
             'basis_function.cpp', 'cluster_list.cpp']

src_files = [cxx_src_folder + x for x in src_files]
src_files.append(cython_folder + 'clease_cxx.pyx')
extra_comp_args = ['-std=c++11']

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
    version=version,
    description="CLuster Expansion in Atomistic Simulation Environment",
    packages=find_packages(),
    download_url='https://gitlab.com/computationalmaterials/clease/-/archive/v{0}/clease-v{0}.zip'.format(version),
    include_package_data=True,
    package_data={'clease.gui': ['layout/*.kv', 'layout/*.png']},
    license='MPL-2.0',
    keywords=['Cluster Expansion', 'Monte Carlo', 'Computational materials', 'Materials research'],
    entry_points={'console_scripts': ['clease=clease.cli.clease:main']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=['ase>=3.18', 'matplotlib', 'spglib', 'scikit-learn']
)
