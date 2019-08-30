# CLEASE

CLuster Expansion in Atomic Simulation Environment

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
CLEASE requires a C++11 compliant compiler available on the system.

# Graphical User Interface
CLEASE comes with a handy graphical user interface (GUI) that can be used to setup Cluster Expansion models.
In order to use the GUI you need to install some additional packages
```
pip install kivy
pip install kivy-garden
garden install matplotlib
```

To launch a GUI simply type
```
clease gui
```
on the command line.

