# CLEASE
| Python | GUI | Average | PyPI | Docs |
| ------ | --- | ------- | ---- | ---- |
| ![coverage](https://gitlab.com/computationalmaterials/clease/badges/master/coverage.svg?job=pytests) | ![coverage](https://gitlab.com/computationalmaterials/clease/badges/master/coverage.svg?job=gui_test) | ![coverage](https://gitlab.com/computationalmaterials/clease/badges/master/coverage.svg) | [![PyPI version](https://badge.fury.io/py/clease.svg)](https://badge.fury.io/py/clease) | [![Documentation Status](https://readthedocs.org/projects/clease/badge/?version=latest)](https://clease.readthedocs.io/en/latest/?badge=latest) |

CLuster Expansion in Atomic Simulation Environment (CLEASE) is a package that automates the cumbersome setup and construction procedure of cluster expansion (CE). It provides a comprehensive list of tools for specifying parameters for CE, generating training structures, fitting effective cluster interaction (ECI) values and running Monte Carlo simulations. A detailed description of the package can be found in the [documentation](https://clease.readthedocs.io/) and our [paper](https://doi.org/10.1088/1361-648X/ab1bbc).

# Partners and Support
![image1](doc/source/resources/image1.png)
![image2](doc/source/resources/image2.png)
![image3](doc/source/resources/image3.png)

# Installation
Install the CLEASE code by executing
```
pip install clease
```
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
CLEASE comes with a handy graphical user interface (GUI) that can be used to setup Cluster Expansion models.
In order to use the GUI you need to install some additional packages
```
pip install kivy
pip install 'https://github.com/kivy-garden/graph/archive/master.zip'
```

There is also a convenience command for doing these two steps when installing CLEASE
```
pip install clease[gui]
```

To launch a GUI simply type
```
clease gui
```
on the command line.

# Troubleshooting

1. If you are running on Mac and get the error

```
fatal error: 'ios' file not found
```

try this before installing

```
export MACOSX_DEPLOYMENT_TARGET=10.14
```

