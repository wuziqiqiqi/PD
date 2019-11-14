# CLEASE
| Python | GUI | Average |
| ------ | --- | ------- |
| ![coverage](https://gitlab.com/computationalmaterials/clease/badges/master/coverage.svg?job=python3_test) | ![coverage](https://gitlab.com/computationalmaterials/clease/badges/master/coverage.svg?job=gui_test) | ![coverage](https://gitlab.com/computationalmaterials/clease/badges/master/coverage.svg) |

CLuster Expansion in Atomic Simulation Environment (CLEASE) is a package that automates the cumbersome setup and construction procedure of cluster expansion (CE). It provides a comprehensive list of tools for specifying parameters for CE, generating training structures, fitting effective cluster interaction (ECI) values and running Monte Carlo simulations. A detailed description of the package can be found in the [user guide](https://computationalmaterials.gitlab.io/clease/) and our [paper](https://doi.org/10.1088/1361-648X/ab1bbc).

# Partners and Support
![image1](doc/source/resources/image1.png)
![image2](doc/source/resources/image2.png)
![image3](doc/source/resources/image3.png)

# Installation
After cloning CLEASE, install the [dependencies](requirements.txt) by executing
```
pip install -r requirements.txt
```
Then install the CLEASE code by executing
```
pip install .
```
in the root folder of the project. If you are a developer you might want to install CLEASE by
```
pip install -e .
```

# Graphical User Interface
CLEASE comes with a handy graphical user interface (GUI) that can be used to setup Cluster Expansion models.
In order to use the GUI you need to install some additional packages
```
pip install kivy
pip install 'https://github.com/kivy-garden/graph/archive/master.zip'
```

There is also a convenience command for doing these two steps
```
clease gui-setup
```

To launch a GUI simply type
```
clease gui
```
on the command line.

