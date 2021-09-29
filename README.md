# CLEASE
| Python | PyPI | Docs |
| ------ | ---- | ---- |
| ![coverage](https://gitlab.com/computationalmaterials/clease/badges/master/coverage.svg) | [![PyPI version](https://badge.fury.io/py/clease.svg)](https://badge.fury.io/py/clease) | [![Documentation Status](https://readthedocs.org/projects/clease/badge/?version=latest)](https://clease.readthedocs.io/en/latest/?badge=latest) |

CLuster Expansion in Atomic Simulation Environment (CLEASE) is a package that automates the cumbersome setup and construction procedure of cluster expansion (CE). It provides a comprehensive list of tools for specifying parameters for CE, generating training structures, fitting effective cluster interaction (ECI) values and running Monte Carlo simulations. A detailed description of the package can be found in the [documentation](https://clease.readthedocs.io/) and our [paper](https://doi.org/10.1088/1361-648X/ab1bbc).

# Installation
Install the CLEASE code by executing
```sh
pip install clease
```

Alternative, CLEASE is also available through anaconda on [conda](https://conda.io) via [conda-forge](https://conda-forge.org/).
We recommend installing CLEASE via conda on windows machines in order to simplify compilations, as pip tends to have
a hard time compiling the C++ code. Install into your conda environment:
```sh
conda install -c conda-forge clease
```

## Development

If you are a developer you might want to install CLEASE by executing the following command in the root folder of the project
```
pip install -e .
```
In order to run the tests, the testing dependencies should be installed. They can be installed with the extra
`test` option
```
pip install .[test]
```
There is an additional option for development purposes, `dev`, which contains some convenience packages.
All of the extras options can be installed via the `all` option,
i.e.
```
pip install .[all]
```
Note, that if you are using `zsh`, you need to [escape the argument](https://stackoverflow.com/a/30539963), e.g.
```
pip install '.[all]'
```

# Graphical User Interface

Clease has a stand-alone jupyter notebook GUI, which is capable of performing most
of the standard CE routines. It can be found [here](https://clease-gui.readthedocs.io).

For simplicity, it can also be installed with `pip install clease[gui]`.

# Troubleshooting

1. If you are running on Mac and get the error

```
fatal error: 'ios' file not found
```

try this before installing

```
export MACOSX_DEPLOYMENT_TARGET=10.14
```

