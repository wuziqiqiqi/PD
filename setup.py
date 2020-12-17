import os
from distutils.sysconfig import get_python_inc
from setuptools import setup, find_packages, Extension


def src_folder():
    candidates = ['cxx/src/', 'clease/', './']
    for c in candidates:
        if os.path.exists(c + 'cluster.cpp'):
            return c
    raise RuntimeError("Cannot find source folder.")


def get_cython_folder():
    candidates = ['cxx/cython/', 'cython/', './']
    for c in candidates:
        if os.path.exists(c + 'clease_cxx.pyx'):
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
about = {}
with open('clease/version.py') as f:
    exec(f.read(), about)
version = about['__version__']

cxx_src_folder = src_folder()
cxx_inc_folder = include_folder()
cython_folder = get_cython_folder()

src_files = [
    'cf_history_tracker.cpp', 'additional_tools.cpp', 'cluster.cpp', 'row_sparse_struct_matrix.cpp',
    'named_array.cpp', 'symbols_with_numbers.cpp', 'basis_function.cpp', 'cluster_list.cpp'
]

src_files = [cxx_src_folder + x for x in src_files]
src_files.append(cython_folder + 'clease_cxx.pyx')
extra_comp_args = ['-std=c++11']

clease_cxx = Extension("clease_cxx",
                       sources=src_files,
                       include_dirs=[
                           cxx_inc_folder,
                           get_npy_include_folder(), cxx_src_folder,
                           get_python_inc(), cython_folder
                       ],
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
    download_url='https://gitlab.com/computationalmaterials/clease/-/archive/v{0}/clease-v{0}.zip'.
    format(version),
    include_package_data=True,
    package_data={'clease.gui': ['layout/*.kv', 'layout/*.png']},
    license='MPL-2.0',
    keywords=['Cluster Expansion', 'Monte Carlo', 'Computational materials', 'Materials research'],
    entry_points={'console_scripts': ['clease=clease.cli.clease:main']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3.6', 'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    install_requires=[
        'ase>=3.19',
        'matplotlib',
        'spglib',
        'scikit-learn',
        'typing_extensions',
        'Deprecated',
        'pytest',
        'pytest-mock',
        'mock',
        # 'mypy',
    ],
    extras_require={
        'dev': (
            'pip',
            'pre-commit',
            'pytest>=4',
            'pytest-mock',
            'ipython',
            'twine',
            'yapf',
            'prospector',
            'pylint',
        ),
        'gui': (
            'kivy>=1.11,<2',
            'kivy-garden.graph @ git+https://github.com/kivy-garden/graph.git@master#egg=kivy-garden.graph'
        )
    },
)
