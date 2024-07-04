import numpy as np
import os
import random

from ase.data import atomic_numbers
from ase.io.lammpsdata import write_lammps_data, read_lammps_data

os.chdir("/Users/Michael_wang/Documents/venkat/mint-PD/PhaseDiagram-Clease/LiKCl-chgnet")

potentialPath = "/jet/home/wuziqi/clease/mint-PD/PhaseDiagram-Clease/Liquid/GAP_potential/gap_LiKCl.xml"

crystalConst = 2.5

numOfT = 26
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
    for T in Ts:
        subDir = myDir + "/" + str(T) + "K"
        os.makedirs(subDir, exist_ok=True)
        
        with open(subDir + "/mixed.in", 'w') as f:
            f.write(f"""read_restart    ../structure.ready

pair_style      chgnet/gpu /nfs/turbo/coe-venkvis/ziqiw-turbo/LAMMPSs/lammps-ASC/potentials/CHGNET
pair_coeff      * * 0.3.0{lmpString}

neighbor        0.3 bin
neigh_modify    delay 10

timestep	    0.001
compute	        msd all msd
thermo_style	custom step temp pe etotal press vol c_msd[4]
thermo	        10
dump	        1 all custom 10 xyz-mix.dump type id x y z
dump	        2 all custom 10 force-mix.dump type id fx fy fz

fix             2 gLiquid npt temp 1500 {T} $(100.0*dt) z 1.01325 1.01325 $(1000.0*dt) couple none
run             10000
unfix           2

fix             3 all npt temp {T} {T} $(100.0*dt) aniso 1.01325 1.01325 $(1000.0*dt)
run             30000
unfix           3

write_data      structure.end.data
write_restart   structure.end
                    """
            )
        
            with open(subDir + "/run-mix.sh", 'w') as f:
                f.write("""#!/bin/bash
#SBATCH -J {name}prep # Job name
#SBATCH -n 4 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH --time=00-6:00:00
#SBATCH -p venkvis-h100
#SBATCH --mem=4096 # Memory pool for all cores in MB
#SBATCH -e prep.err #change the name of the err file 
#SBATCH -o prep.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=BEGIN,END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent

module load spack/0.21
unset SPACK_DISABLE_LOCAL_CONFIG
module load anaconda3

source /nfs/turbo/coe-venkvis/ziqiw-turbo/.bashrc

conda activate chgnet

echo "Job started on `hostname` at `date`" 

/nfs/turbo/coe-venkvis/ziqiw-turbo/LAMMPSs/lammps-ASC/build/lmp -in mixed.in > mixed.log

echo " "
echo "Job Ended at `date`"
        """.format(name=myDir[:4] + "-" + str(T)))
        
print("Step1 done!")
    