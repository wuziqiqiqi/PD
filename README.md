# Phase Diagram Calculator

Currently, this phase digram calculator only support calculations in solidus region. Support for liquidus region will be added later

## Installation
First create a environment with python=3.8.x

```
conda create --name <name> python=3.8
conda activate <name>

```

Then, clone code with:

```
git clone https://github.com/rectify-mint/mint-PD.git
```

Then run:

```
cd mint-PD/PhaseDiagram-Clease
pip install .[all]
```

If you are using `zsh`, you need to run:

```
pip install '.[all]'
```
In the end, install tqdm

```
pip install tqdm
```


This should get you all set on python packages.

***If you plan to rely solely on DFT, you can skip the all the following***

To install LAMMPS, run: (make sure you are in the correct python environment)

```
git clone -b stable_29Sep2021_update2 --depth 1 https://github.com/lammps/lammps.git
cd lammps && mkdir build && cd build
cmake -C ../cmake/presets/basic.cmake -D PKG_MC=on -D PKG_MEAM=on -D PKG_MOLECULE=on -D BUILD_SHARED_LIBS=on -D LAMMPS_EXCEPTIONS=on -D PKG_PYTHON=on -D PKG_MANYBODY=ON -D PKG_MISC=ON -D PKG_EXTRA-COMPUTE=ON -D PKG_EXTRA-DUMP=ON -D PKG_EXTRA-FIX=ON -D PKG_EXTRA-PAIR=ON -D BUILD_MPI=ON -D BUILD_OMP=yes -D PKG_MPIIO=ON ../cmake
cmake --build .
make install-python
```

To verify if LAMMPS can be successfully started from Python, start the Python interpreter, load the lammps Python module and create a LAMMPS instance. This should not generate an error message and produce output similar to the following:

```
$ python
Python 3.8.5 (default, Sep  5 2020, 10:50:12)
[GCC 10.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import lammps
>>> lmp = lammps.lammps()
LAMMPS (18 Sep 2020)
using 1 OpenMP thread(s) per MPI task
>>>
```

## Usage

There are three main part in computationally plotting phase diagram:

1.	Get energy calculator ready (In this example, this will be fitting a Cluster Expansion (CE) model with CE training3.ipynb.)
2. Setups for phase boundray tracing algorithm (running emc.py)
3. Running the phase boundray tracing algorithm (phb.py)

If you go into PhaseDiagram-Clease directory, you'll find the `example.yaml` file. This is where a user specify all his/her input parameters. CLEASE, LAMMPS, DFT are sections for fitting CE models. EMC section is for emc.py, and PHB section is for phb.py.

To run LiNa example, open the `example.yaml`. If you decide to use LAMMPS, comment out the DFT section. Otherwise, comment-out the LAMMPS section and enter the name of the DFT energy file, a `.db` file that contains both structures of interests and their corresponding energy. Everything else in `example.yaml` should be good. You simply need to run CE training3.ipynb, emc.py, phb.py in sequence.


