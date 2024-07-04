import json
import logging
import sys, os, time
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from lammps import lammps

from ase.db import connect
from ase.calculators.emt import EMT
from ase.calculators.eam import EAM
from ase.calculators.lammpslib import LAMMPSlib
from ase.visualize import view
from ase import Atom, Atoms
from ase.build import bulk
from ase.data import atomic_numbers
from ase.io.trajectory import TrajectoryWriter
from ase.io import write, read

from clease.settings import Concentration
from clease.structgen import NewStructures
from clease.settings import CEBulk
from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo, constraints, SGCMonteCarlo, BinnedBiasPotential, MetaDynamicsSampler
from clease.montecarlo.observers import ConcentrationObserver, MoveObserver, CorrelationFunctionObserver
from clease.montecarlo.trial_move_generator import RandomTransition
from clease.tools import update_db
from clease import Evaluate
import clease.plot_post_process as pp

from drawpd import LTE, MF

kB = 8.617333262e-5
np.set_printoptions(precision=3, suppress=True)

"""
⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽
############# DEBUG == 0 #############
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Z test acc
phi_LTE
normal scan:
    T, mu
    convergence progress: i = ?000, coverged after ?

⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽
############# DEBUG == 1 #############
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
normal scan:
    print convergerd atoms to termial
    array of qautities used for disconti test [oldYs, newY]

disconti_test:
    fitted function
    best order of fit
    new predictions
    little v
    sigma
    @ what mu, z score = ?

⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽
############# DEBUG == 2 #############
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
normal scan:
    PLOT converging history

disconti_test:
    PLOT data, fit, and predictions

"""
with open('Liquid/LiMg-liquid-example.yaml', 'r') as file:
    options = yaml.safe_load(file)


DEBUG = options["EMC"]["DEBUG"]

if "LAMMPS" in options:
        from ase.calculators.lammpslib import LAMMPSlib
        print("LAMMPS calc!")
        ASElammps = LAMMPSlib(**options["LAMMPS"])


GroundStates = options["GroundStates"]
gs_db_names = options["EMC"]["gs_db_names"]
muInit, muFinal = options["EMC"]["muInit"], options["EMC"]["muFinal"]
dMu = options["EMC"]["dMu"]

dMu = abs(dMu)
if (muFinal - muInit)*dMu < 0:
    dMu = -dMu
    
for gsIdx, gs_name in enumerate(GroundStates):
    db_name = options["CLEASE"][gs_name]["CESettings"]["db_name"]
    conc = Concentration(basis_elements=options["CLEASE"][gs_name]["CESettings"]["concentration"])

    tmp = basis_elements=options["CLEASE"][gs_name]["CESettings"].copy()
    tmp['concentration'] = conc
    MCsettings = CEBulk(**tmp)

    eciName = options["CLEASE"][gs_name]["CEFitting"]["ECI_filename"]
    if eciName == "FROM DB":
        with open(db_name + "-eci.json") as f:
            eci = json.load(f)
    else:
        with open(eciName) as f:
            eci = json.load(f)

    # atoms = MCsettings.prim_cell.copy()
    # atoms *= (4,4,4)
    # atoms.write(options["EMC"]["gs_db_names"][0])

    # atoms = MCsettings.prim_cell.copy()
    # atoms.numbers[0] = atomic_numbers[options["CLEASE"][gs_name]["CESettings"]["concentration"][0][1]]
    # atoms *= (4,4,4)
    # atoms.write(options["EMC"]["gs_db_names"][1])

    db = connect(gs_db_names[gsIdx])
    gs05 = None
    for row in db.select(""):
        gs05 = row.toatoms()

    # gs05 = attach_calculator(MCsettings, atoms=gs05, eci=eci)
    gs05.calc = ASElammps
    print(gs05.get_potential_energy())
    
    TInit, TFinal = options["EMC"]["TInit"], options["EMC"]["TFinal"]
    dT = options["EMC"]["dT"]
    Ts = np.arange(TInit, TFinal+1e-5, dT)
    # Ts = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    if gsIdx == 0:
        mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)
    elif gsIdx == 1:
        muTmp = muInit
        muInit = muFinal
        muFinal = muTmp
        dMu = -dMu
        mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)
    
    
    for temp in Ts:
        print(temp)
        for myProgress in range(100):
            print(myProgress)
            mc = Montecarlo(gs05, temp, generator=RandomTransition(gs05))
            # mc = Montecarlo(gs05, temp)
            mc.run(216)
            write("Liquid/LiMg-Liquid.xyz", gs05, append=True)
    
    break
