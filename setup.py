import os
from pathlib import Path
import sysconfig
from setuptools import setup, Extension


def src_folder():
    candidates = ["cxx/src/", "clease/", "./"]
    for c in candidates:
        if os.path.exists(os.path.join(c, "cluster.cpp")):
            return c
    raise RuntimeError("Cannot find source folder.")


def get_cython_folder():
    candidates = ["cxx/cython/", "cython/", "./"]
    for c in candidates:
        if os.path.exists(os.path.join(c, "clease_cxx.pyx")):
            return c
    raise RuntimeError("Cannot find cython folder")


def include_folder():
    candidates = ["cxx/include/", "clease/", "./"]
    for c in candidates:
        if os.path.exists(os.path.join(c, "cluster.hpp")):
            return c
    raise RuntimeError("Cannot find include folder")


def get_npy_include_folder():
    import numpy as np

    return np.get_include()


def build_ext(ext_module):
    from Cython.Build import cythonize

    return cythonize(
        ext_module, compiler_directives={"language_level": "3", "embedsignature": True}
    )


cxx_src_folder = src_folder()
cxx_inc_folder = include_folder()
cython_folder = get_cython_folder()


def get_source_files():
    """Construct a list of all sources to be included."""
    cpp_sources = Path(cxx_src_folder).glob("*.cpp")

    # C++ sources, excluding the ones which rely on NumPy
    # those are included directly from the Cython source
    sources = [str(f) for f in cpp_sources]

    # Cython sources
    cython_src = ["clease_cxx.pyx"]
    for src in cython_src:
        sources.append(os.path.join(cython_folder, src))
    return sources


src_files = get_source_files()
extra_comp_args = ["-std=c++11"]
# Replaces the deprecated "distutils.sysconfig.get_python_inc()"
py_include = sysconfig.get_path("include")

clease_cxx = Extension(
    "clease_cxx",
    sources=src_files,
    include_dirs=[
        cxx_inc_folder,
        get_npy_include_folder(),
        cxx_src_folder,
        py_include,
        cython_folder,
    ],
    extra_compile_args=extra_comp_args,
    language="c++",
)

# Extra packages required with certain aspects of CLEASE, which are not required
# to run the base CLEASE package.
EXTRAS_REQUIRE = {
    # For building the documentation
    "doc": ("sphinx", "sphinx_rtd_theme"),
    # For running the CLEASE test suite
    "test": (
        "pytest",
        "pytest-mock",
        "mock",
        # Get the histrogram extras, for making nice histogram plots
        # with pytest-benchmark
        "pytest-benchmark[histogram]>=3.4.1",
        "tox>=3.24.0",
    ),
    # Extra nice-to-haves when developing CLEASE
    "dev": (
        "pip",
        "cython",
        "pre-commit",
        "ipython",
        "twine",
        "black>=22.1.0",  # Style formatting
        "pylint",
        "pyclean>=2.0.0",  # For removing __pycache__ and .pyc files
        "pytest-cov",
        "build",
    ),
    "gui": ("clease-gui",),
}
# Make an entry which installs all of the above in one go
EXTRAS_REQUIRE["all"] = tuple({package for key, tup in EXTRAS_REQUIRE.items() for package in tup})

setup(
    ext_modules=build_ext(clease_cxx),
    extras_require=EXTRAS_REQUIRE,
)
