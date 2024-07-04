import numpy as np
import os
import shutil
import glob
from clease.settings import Concentration
from clease.settings import CEBulk
from clease.structgen import NewStructures
from ase.db import connect
from ase.io import read as ase_read
from ase.calculators.eam import EAM
from ase.visualize import view
from clease.tools import update_db
from clease import Evaluate
from clease import NewStructures
from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo
import clease

# from gpaw import GPAW
from ase.calculators.emt import EMT
import json

import clease.plot_post_process as pp
import matplotlib.pyplot as plt

# from KianCalculator import KianCalc

from ase.calculators.lammpslib import LAMMPSlib

cmds = ["pair_style meam",
        "pair_coeff * * /Users/Michael_wang/Documents/venkat/MEAM/library.meam Li Mg /Users/Michael_wang/Documents/venkat/MEAM/LiMg.meam Li Mg"]
rootDir = "/Users/Michael_wang/Documents/venkat/cleaseASEcalc/LiMg"
ASElammps = LAMMPSlib(lmpcmds=cmds, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSlogMg.log"))

# cmds = ["pair_style eim",
#         "pair_coeff * * Na Li /Users/Michael_wang/Documents/venkat/cleaseASEcalc/ffield.eim Na Li"]
# rootDir = "/Users/Michael_wang/Documents/venkat/cleaseASEcalc/LiNa"
# ASElammps = LAMMPSlib(lmpcmds=cmds, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSlog.log"))

db_name = "LiMg/LiMg-Sep8-first-Batch-bcc.db"
# db_name = "LiNa/LiNa-Sep5-first-Batch-EOS-7-cutoff.db"


dbTmp = connect(db_name)
for idx, row in enumerate(dbTmp.select('')):
    if idx < 2:
        continue
    atoms = row.toatoms()
    atoms.calc = ASElammps
    # view(atoms)
    atoms.numbers[0] = 3
    s = atoms.get_chemical_symbols()
    print(s)
    print(atoms.get_potential_energy())
    break

exit(0)

# def find_final_structure(dir, formula):
#     #### for a folder of trajs ####
#     # if formula in os.listdir(dir):
#     #     pth = glob.glob(os.path.join(dir, formula, "*"))[0]
#     #     traj = ase_read(pth, index=":")
#     #     return traj[-1]
#     # else:
#     #     return None
#
#     #### for db file ####
#     myatom = None
#     db = connect(dir)
#     for row in db.select(formula=formula):
#         myatom = row.toatoms()
#     return myatom

gen_num = None

# np.random.seed(0)  # Set a seed for consistent tests
conc = Concentration(basis_elements=[['Li', 'Na']])

db_name = "LiNa-Aug4-fuck.db"

genNumFileName = os.path.abspath(db_name) + ".genNum.txt"
print(genNumFileName)
try:
    with open(genNumFileName, "r") as f:
        gen_num = f.read()
except:
    gen_num = 999999

assert gen_num, "gen_num init failed"

print(gen_num)


settings = CEBulk(crystalstructure='bcc',
                  a=3.51,
                  supercell_factor=80,
                  concentration=conc,
                  db_name=db_name,
                  max_cluster_dia=[4.0, 4.0, 4.0],
                  basis_func_type="polynomial")

# ns = NewStructures(settings, generation_number=gen_num, struct_per_gen=2)
# ns.generate_probe_structure()
# exit()

template = settings.prim_cell.copy()
template *= (5,4,4)

for i in range(0,20):
    ns = NewStructures(settings, generation_number=gen_num, struct_per_gen=1)
    ns.generate_random_structures(atoms=template, conc=np.array([i/20, 1-i/20]))
    gen_num += 1

exit()

db = connect(db_name)
with open('aucu-Mar17.db-eci.json') as f:
    eci = json.load(f)
iii = 0
skipped = 0
for row in db.select(''):
    print("woring on:", iii)
    template = row.toatoms()
    try:
        ns = NewStructures(settings, generation_number=gen_num, struct_per_gen=1)
        ns.generate_gs_structure(atoms=template, init_temp=3000,
                                 final_temp=2000, num_temp=200,
                                 num_steps_per_temp=5000,
                                 eci=eci, random_composition=True)
        gen_num += 1
    except:
        skipped += 1
        print(skipped, "structure skipped")
    iii += 1

# MUST CALL: if calc.has_energy(row.formula) before each get_potential_energy !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# calc = KianCalc("CoaseE14.json")
calc = EMT()
# calc = EAM(potential='AuCu_Zhou04.eam.alloy')


# from ase.optimize import BFGS
# from ase.constraints import UnitCellFilter
#
# db = connect(db_name)
#
# # Run calculations for all structures that are not converged.
# for idx, row in enumerate(db.select(converged=False)):
#     print("working on ", idx)
#     atoms = row.toatoms()
#     atoms.calc = calc
#     # ucf = UnitCellFilter(atoms)
#     opt = BFGS(atoms, logfile=None)
#     opt.run(fmax=0.02)
#     custom_kvp = row.key_value_pairs.copy()
#     for key in ("struct_type", "name", "converged", "started", "queued"):
#         custom_kvp.pop(key, None)
#     update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name, custom_kvp_final=custom_kvp)


db = connect(db_name)
calculatedNum = 0

for row in db.select(converged=False):
    atoms = row.toatoms()
    atoms.calc = calc
    atoms.get_potential_energy()
    update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)
    calculatedNum += 1
    print(calculatedNum)

# for row in db.select(converged=False):
#     atoms = row.toatoms()
#     atoms.calc = calc
#     if calc.has_energy(row.formula):
#         atoms.get_potential_energy()
#         update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)

# for row in db.select(converged=False):
#     if row.formula != "Au32Cu10":
#         finalStrut = find_final_structure("/Users/Michael_wang/Documents/venkat/clease/db_test/aucu_spe.db", row.formula)
#         if finalStrut:
#             print(finalStrut.calc.results['energy'])
#             update_db(uid_initial=row.id, final_struct=finalStrut, db_name=db_name)

# clease.reconfigure(settings)

# with open(genNumFileName, "w") as f:
#     f.write(str(gen_num))

eva = Evaluate(settings=settings, scoring_scheme='k-fold', nsplits=10)
eva.set_fitting_scheme(fitting_scheme='l1')

alpha = eva.plot_CV(alpha_min=1E-7, alpha_max=1.0, num_alpha=50)

# set the alpha value with the one found above, and fit data using it.
eva.set_fitting_scheme(fitting_scheme='l2', alpha=alpha)
eva.fit()  # Run the fit with these settings.

fig = pp.plot_fit(eva)
plt.show()

fig = pp.plot_convex_hull(eva)
plt.show()


# plot ECI values
fig = pp.plot_eci(eva)
plt.show()

eva.save_eci(fname=db_name + "-eci.json")