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
    return cythonize(ext_module, compiler_directives={"language_level": "3"})


# Get version number
about = {}
with open(os.path.join('clease', 'version.py')) as f:
    exec(f.read(), about)
version = about['__version__']

cxx_src_folder = src_folder()
cxx_inc_folder = include_folder()
cython_folder = get_cython_folder()

src_files = [
    'cf_history_tracker.cpp',
    'additional_tools.cpp',
    'cluster.cpp',
    'row_sparse_struct_matrix.cpp',
    'named_array.cpp',
    'symbols_with_numbers.cpp',
    'basis_function.cpp',
    'cluster_list.cpp',
    'atoms.cpp',
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

# Extra packages required with certain aspects of CLEASE, which are not required
# to run the base CLEASE package.
EXTRAS_REQUIRE = {
    # For building the documentation
    'doc': ('sphinx', 'sphinx_rtd_theme'),
    # For running the CLEASE test suite
    'test': (
        'pytest',
        'pytest-mock',
        'mock',
    ),
    # Extra nice-to-haves when developing CLEASE
    'dev': (
        'pip',
        'cython',
        'pre-commit',
        'ipython',
        'twine',
        'yapf',
        'pylint',
        'pyclean>=2.0.0',  # For removing __pycache__ and .pyc files
        'tox~=3.24.0',
        'pytest-cov',
        'build',
    ),
    'gui': ('clease-gui',),
}
# Make an entry which installs all of the above in one go
# Separate out "gui" from "all"
EXTRAS_REQUIRE['all'] = tuple(
    {package for key, tup in EXTRAS_REQUIRE.items() for package in tup if key != 'gui'})

setup(
    name="clease",
    ext_modules=build_ext(clease_cxx),
    author=['J. H. Chang', 'D. Kleiven', 'A. S. Tygesen'],
    author_email="jchang@dtu.dk, david.kleiven@ntnu.no, alexty@dtu.dk",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://gitlab.com/computationalmaterials/clease',
    version=version,
    description="CLuster Expansion in Atomistic Simulation Environment",
    packages=find_packages(),
    download_url='https://gitlab.com/computationalmaterials/clease/-/archive/v{0}/clease-v{0}.zip'.
    format(version),
    include_package_data=True,
    license='MPL-2.0',
    project_urls={
        'Documentation': 'https://clease.readthedocs.org/',
        'Source': 'https://gitlab.com/computationalmaterials/clease/',
    },
    keywords=['Cluster Expansion', 'Monte Carlo', 'Computational materials', 'Materials research'],
    entry_points={'console_scripts': ['clease=clease.cli.main:clease_cli']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'ase>=3.20',
        'numpy',
        'cython',
        'matplotlib',
        'spglib',
        'scikit-learn',
        'typing_extensions',
        'Deprecated',
        'click>=8.0.0',  # CLI things
        'attrs',
        'scipy>=1.5.0',  # Last version which allows python 3.6
        # 'mypy',
    ],
    extras_require=EXTRAS_REQUIRE,
)
