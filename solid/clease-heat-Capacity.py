import numpy as np
from clease.settings import Concentration
from clease.structgen import NewStructures
from clease.settings import CEBulk
from ase.db import connect
import json
from ase.calculators.emt import EMT
from ase.calculators.eam import EAM
from ase.calculators.lammpslib import LAMMPSlib
from ase.db import connect
from ase.visualize import view
from ase import Atom, Atoms
from ase.build import bulk

from clease.tools import update_db

from clease import Evaluate
import clease.plot_post_process as pp
import matplotlib.pyplot as plt

from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo, constraints, SGCMonteCarlo, BinnedBiasPotential, MetaDynamicsSampler
from clease.montecarlo.observers import ConcentrationObserver

import logging
import sys, os

from lammps import lammps

consIdx = sys.argv[1]
conss =  np.linspace(0.3, 0.5, 10)
# cons = conss[int(consIdx)]

rootDir = 'ConsIdx'+consIdx

logging.basicConfig(filename=os.path.join(rootDir, 'CLEASE-heatCap.log'), level=logging.INFO)
np.random.seed(42)  # Set a seed for consistent tests


cmds = ["pair_style eam/alloy",
        "pair_coeff * * AuCu_Zhou04.eam.alloy Cu Au"]

# Ni = bulk('Ni', cubic=True)
# H = Atom('H', position=Ni.cell.diagonal()/2)
# NiH = Ni + H

ASElammps = LAMMPSlib(lmpcmds=cmds, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSlog.log"), mpi_command='mpirun -np 4')
# lmp = lammps()


# print(ASElammps, lmp)

# exit(0)

conc = Concentration(basis_elements=[['Au', 'Cu']])

curr_db_name = "aucuEAMUnrelaxed.db"

MCsettings = CEBulk(crystalstructure='fcc',
                  a=3.8,
                  supercell_factor=27,
                  concentration=conc,
                  db_name=curr_db_name,
                  max_cluster_dia=[6.0, 4.5, 4.5])

atoms = MCsettings.prim_cell.copy()
atoms *= (6,6,6)
MCsettings.set_active_template(atoms)

# atoms.calc = lammps


# print(atoms)

# prefix = 'ASE-EAM-aucu_metadyn-'

heat_caps = []
critical_temps = []
for cons in conss:

    x = np.array([cons, 1-cons])
    print("############# cons = ", x, "##############")


    # In[14]:

    new_struct = NewStructures(MCsettings)
    num_atoms_in_basis = [len(indices) for indices in MCsettings.index_by_basis]
    print(num_atoms_in_basis)


    num_to_insert = MCsettings.concentration.conc_in_int(num_atoms_in_basis, x)
    print(num_to_insert)


    MCns = new_struct._random_struct_at_conc(num_to_insert)
    print(MCns)


    # with open('eci_l1.json') as f:
    #     eci = json.load(f)
    # MCns = attach_calculator(MCsettings, atoms=MCns, eci=eci)

    MCns.calc = ASElammps

    print(MCns)

    t_max = 1000
    t_min = 20
    t_steps = 100
    num_sweep = 700
    Ts = np.logspace(np.log10(t_max), np.log10(t_min), t_steps)
    num_step = num_atoms_in_basis[0] * num_sweep
    mc_data = []
    for ii, temp in enumerate(Ts):
        print(temp)
        mc = Montecarlo(MCns, temp)
        cnst = constraints.ConstrainSwapByBasis(mc.atoms, MCsettings.index_by_basis)
        mc.generator.add_constraint(cnst)
        mc.run(steps=num_step)
        mc_data.append(mc.get_thermodynamic_quantities())
        # print(mc.get_thermodynamic_quantities()['heat_capacity'])

    y = []
    for data in mc_data:
        y.append(data['heat_capacity'])

    heat_caps.append(y)
    critical_temps.append(Ts[np.argmax(y)])

# np.save(os.path.join(rootDir, "yyyyyyyyiyiyiyiy.npy"), np.array(y))
np.save("LAMMPSCdata.npy", np.array(heat_caps))
# from matplotlib import pyplot as plt
# plt.plot(conss, critical_temps)
# plt.show()
#
# plt.imshow(np.array(heat_caps))
# plt.show()