import sys
sys.path.insert(0, '/ocean/projects/cts180021p/wuziqi/clease/mint-PD/PhaseDiagram-Clease')

import numpy as np
import os
import random

from ase.data import atomic_numbers
from ase.io.lammpsdata import read_lammps_data
from lammpsdata import write_lammps_data

potentialPath = "/ocean/projects/cts180021p/wuziqi/clease/mint-PD/PhaseDiagram-Clease/Liquid/GAP_potential/gap_LiKCl.xml"

crystalConst = 2.5

numOfT = 25
TInit, Tfinal = 300, 1200
Ts = np.linspace(TInit, Tfinal, numOfT)

myDirs = [x for x in os.listdir() if x != 'step1.py' and x != '.DS_Store']
print(myDirs)

for myDir in myDirs:
    with open(myDir + "/123toABC.txt", 'r') as f:
        lmpString = f.readline()
        # [1:] to discharge the leading '': ['', 17, 19, 3]
        lmpAtomConversion = lmpString.split(' ')[1:]
        print("lmpAtomConversion", lmpAtomConversion)
        
    structureA = read_lammps_data(myDir + "/structureA.data")
    for idx, i in enumerate(lmpAtomConversion):
        structureA.numbers[structureA.numbers == idx+1] = int(i)
        
    structureB = read_lammps_data(myDir + "/structureB.data")
    for idx, i in enumerate(lmpAtomConversion):
        structureB.numbers[structureB.numbers == idx+1] = int(i)
        
    zmin = structureB.positions[:, 2].min()
    zmax = structureA.positions[:, 2].max()
    delta = zmax - zmin + crystalConst
    
    structureB.positions += (0, 0, delta)
    finalAtoms = structureA + structureB
    
    zmin = structureA.positions[:, 2].min()
    zmax = structureB.positions[:, 2].max()
    delta = zmax - zmin + crystalConst
    finalAtoms.set_cell([finalAtoms.cell[0,0], finalAtoms.cell[1,1], delta], scale_atoms=False)
    
    for T in Ts:
        subDir = myDir + "/" + str(T) + "K"
        os.makedirs(subDir, exist_ok=True)
        write_lammps_data(file=subDir + "/mixed.data", atoms=finalAtoms, atom_style='full')
        
        with open(subDir + "/mixed.in", 'w') as f:
            f.write("units\tmetal\n")
            f.write("atom_style\tfull\n")
            f.write("\n")
            f.write("read_data\tmixed.data\n")
            f.write("\n")
            f.write("pair_style\tquip\n")
            f.write("pair_coeff\t* * " + potentialPath + " \"Potential xml_label=GAP_2021_9_27_0_1_17_37_618\"" + lmpString + "\n")
            f.write("\n")
            f.write("neighbor\t0.3 bin\n")
            f.write("neigh_modify\tdelay 10\n")
            f.write("\n")
            f.write("timestep\t0.001\n")
            f.write("compute\tmsd all msd\n")
            f.write("thermo_style\tcustom step temp pe etotal press vol c_msd[4]\n")
            f.write("thermo\t10\n")
            f.write("dump\t1 all custom 10 xyz-mix.dump type id x y z\n")
            f.write("dump\t2 all custom 10 force-mix.dump type id fx fy fz\n")
            f.write(
                """
fix     1 all nve
run     2000
unfix   1

velocity    all create {T} 1312343 dist gaussian
fix     2 all npt temp {T} {T} $(100.0*dt) aniso 0.0 1.01325 $(1000.0*dt)
run     3000
unfix   2

fix     3 all npt temp {T} {T} $(100.0*dt) aniso 1.01325 1.01325 $(1000.0*dt)
run     20000
unfix   3
                """.format(T=T)
            )
        
        with open(subDir + "/run-mixed.sh", 'w') as f:
            f.write(
                """#!/bin/bash
#SBATCH -J {name} # Job name
#SBATCH -n 16 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --time=00-12:00:00
#SBATCH -A cts180021p
#SBATCH -p RM-shared
#SBATCH --mem-per-cpu=2000 # Memory pool for all cores in MB
#SBATCH -e mixed.err #change the name of the err file 
#SBATCH -o mixed.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent

module load intelmpi/2021.3.0-intel2021.3.0

echo "Job started on `hostname` at `date`" 

mpirun -np $SLURM_NTASKS /ocean/projects/cts180021p/wuziqi/clease/lammps-stable_29Sep2021_update3/GAPbuild/lmp -in mixed.in > mixed.log

echo " "
echo "Job Ended at `date`"
                """.format(name=myDir[:4] + "-" + str(T))
            )

print("Step1 done!")
    