import numpy as np
import os
import shutil
import glob
from clease.settings import Concentration
from clease.settings import CEBulk
from clease.structgen import NewStructures
from ase.db import connect
from ase.io import read as ase_read
from ase.visualize import view
from ase.calculators.eam import EAM
from clease.tools import update_db
from clease import Evaluate
from clease import NewStructures
from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo
import clease

from ase.optimize import BFGS
from ase.constraints import UnitCellFilter, ExpCellFilter, StrainFilter

# from gpaw import GPAW
from ase.calculators.emt import EMT
import json

import clease.plot_post_process as pp
import matplotlib.pyplot as plt

from ase.calculators.lammpslib import LAMMPSlib

cmds0 = ["pair_style eim",
        "pair_coeff * * Na /Users/Michael_wang/Documents/venkat/cleaseASEcalc/ffield.eim Na"]
rootDir = "/Users/Michael_wang/Documents/venkat/cleaseASEcalc/LiNa"
ASElammps0 = LAMMPSlib(lmpcmds=cmds0, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSlogNa.log"))

cmds1 = ["pair_style eim",
        "pair_coeff * * Li /Users/Michael_wang/Documents/venkat/cleaseASEcalc/ffield.eim Li"]
rootDir = "/Users/Michael_wang/Documents/venkat/cleaseASEcalc/LiNa"
ASElammps1 = LAMMPSlib(lmpcmds=cmds1, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSlogLi.log"))

cmds = ["pair_style eim",
        "pair_coeff * * Na Li /Users/Michael_wang/Documents/venkat/cleaseASEcalc/ffield.eim Na Li"]
rootDir = "/Users/Michael_wang/Documents/venkat/cleaseASEcalc/LiNa"
ASElammps = LAMMPSlib(lmpcmds=cmds, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSlog.log"))


import re, numpy

def get_conc_from_formula(formula):
    sperated = re.split('Li|Na', formula)
    for i in range(len(sperated)):
        if sperated[i] == '':
            sperated[i] = '1'
    if len(sperated) == 3:
        return int(sperated[1])/(int(sperated[2]) + int(sperated[1]))
    elif len(sperated) == 2:
        if formula[:2] == "Li":
            return 1
        elif formula[:2] == "Na":
            return 0
    return numpy.inf

with open('LiNa/LiNa-merged.db-eci.json') as f:
    eci = json.load(f)

x = []
E = []

db2run = connect("LiNa/LiNa-Sep1-first-Batch-3.9.db")
db2result = connect("LiNa/LiNa-asasfdascvzxcvas.db")
for iii, row in enumerate(db2run.select('')):
    if iii == 0:
        continue
    print("working on", iii)
    fomula = row.formula
    x.append(get_conc_from_formula(fomula))
    scale = (3.51*(x[-1]) + 4.29*(1-x[-1]))/3.9
    atoms = row.toatoms()
    atoms.cell *= scale
    atoms.positions *= scale
    # atoms = attach_calculator(hhsettings, atoms=atoms, eci=eci)
    if x[-1] == 0:
        atoms.calc = ASElammps0
    elif x[-1] == 1:
        atoms.calc = ASElammps1
    else:
        atoms.calc = ASElammps

    # ucf = StrainFilter(atoms, mask=[True,True,True,True,True,True])
    # opt = BFGax=0.00001)
    E.append(atoms.get_potential_energy())
    db2result.write(atoms)

a = np.array(sorted(zip(x, E)))
x = a[:,0]
E = a[:,1]

Na = E[0]
Li = E[-1]

for i in range(len(E)):
    E[i] = (E[i] - (x[i])*Li - (1-x[i])*Na)

plt.plot(x,E, "o--")
plt.title("Toy Model Formation Energy")
plt.ylabel("eV/atom")
plt.xlabel("x")
plt.show()