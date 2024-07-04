import numpy as np
import os
import random

from ase.data import atomic_numbers, atomic_masses
from ase.io.lammpsdata import write_lammps_data
from ase.visualize import view
from ase.io import read, write
from ase import Atoms

from chgnet.model.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics, StructOptimizer
from pymatgen.core import Structure
import warnings
import os

from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

mpr = MPRester("X95IHGgtNwjR1xILRRk9NImcElr5JlTH")

def distance_matrix(points):
    # Convert the list of points to a NumPy array
    points_array = np.array(points)
    
    # Compute the squared Euclidean distances using broadcasting
    diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    
    return dist_matrix

numOfX = 11
initialTemp = "100.0"
alloySystem = "LiKCl-chgnet-0.3.0-relaxed"
potentialPath = "/ocean/projects/cts180021p/wuziqi/clease/mint-PD/PhaseDiagram-Clease/Liquid/GAP_potential/gap_LiKCl.xml"
chgnetDriverPath = "/nfs/turbo/coe-venkvis/ziqiw-turbo/LAMMPSs/lammps-ASC/potentials/CHGNET"
model_name = "0.3.0"
# model_name = "/ocean/projects/cts180021p/wuziqi/clease/mace/mace/calculators/foundations_models/2023-12-03-mace-mp.model-lammps.pt"
# model_name = "path /nfs/turbo/coe-venkvis/ziqiw-turbo/epoch999_e52_f152_s406_mNA.pth.tar"
# model_name = "/ocean/projects/cts180021p/wuziqi/clease/mint-PD/PhaseDiagram-Clease/Liquid/GAP_potential/gap_LiKCl.xml \"Potential xml_label=GAP_2021_9_27_0_1_17_37_618\""
multipicity = (4,4,12)

pair_style  = f"chgnet/gpu {chgnetDriverPath}"
# pair_style  = f"quip"
# pair_style  = f"mace no_domain_decomposition"
pair_coeff  =  f"* * {model_name}"

end0StructureMPid = 'mp-22905' # a = 2.55 2.46
end0StructureA = 2.55
# end1StructureMPid = 'mp-22983'
end1StructureMPid = 'mp-23193' # a = 3.18 
end1StructureA = 3.18


structure = mpr.get_structure_by_material_id(end0StructureMPid)
sg_analyzer = SpacegroupAnalyzer(structure)
primitive_structure = sg_analyzer.get_conventional_standard_structure()
ase_atoms0 = AseAtomsAdaptor.get_atoms(primitive_structure)
cell0 = ase_atoms0.cell
ase_atoms0.set_cell(cell0.cellpar()[:3]*2*end0StructureA/cell0.cellpar()[0], scale_atoms=True)
view(ase_atoms0)

structure = mpr.get_structure_by_material_id(end1StructureMPid)
sg_analyzer = SpacegroupAnalyzer(structure)
primitive_structure = sg_analyzer.get_conventional_standard_structure()
ase_atoms1 = AseAtomsAdaptor.get_atoms(primitive_structure)
cell1 = ase_atoms1.cell
ase_atoms1.set_cell(cell1.cellpar()[:3]*2*end1StructureA/cell1.cellpar()[0], scale_atoms=True)
view(ase_atoms1)

scaledCell = np.linspace(cell0*multipicity, cell1*multipicity, 10001, endpoint=True)
xs = np.linspace(0, 1, numOfX, endpoint=True)

# scaledCell = np.linspace(cell0, cell1, 10001, endpoint=True)
# number of unit cell in the base layer will change species, out of multipicity[0] * multipicity[1]
# ns = np.linspace(0, multipicity[0] * multipicity[1], numOfX, endpoint=True)
print(xs)

for idx, xxx in enumerate(xs):
    subDir = alloySystem + "/{:.2f}x-".format(xxx) + alloySystem
    os.makedirs(subDir, exist_ok=True)
    if idx == numOfX - 1:
        tmpAtoms = (ase_atoms1*multipicity).copy()
    elif idx == 0:
        tmpAtoms = (ase_atoms0*multipicity).copy()
    else:
        ##################################### phase seperate #####################################
        # if 1:
        #     # Define the supercell dimensions
        #     tmpAtomsDims = list(multipicity)
        #     tmpAtomsDims[-1] = 1

        #     # Create a new Atoms object to store the supercell
        #     tmpAtoms = Atoms()

        #     NSpecies2Change = xxx

        #     count = 0
        #     toBeAdded = np.zeros((3,3))
            
        #     # Define the supercell dimensions
        #     supercellCellSize = np.zeros((3,3))
        #     supercellCellSize = ase_atoms0.cell * [0,tmpAtomsDims[1],1]

        #     origin = np.array([0.0,0.0,0.0])

        #     # Loop over the supercell dimensions and append copies of the unit cell
        #     for i in range(tmpAtomsDims[0]):
        #         for j in range(tmpAtomsDims[1]):
        #             # Create a copy of the unit cell
        #             cell_copy = ase_atoms0.copy()
                    
        #             # Decide whether change species 
        #             if count < NSpecies2Change:
        #                 cell_copy.numbers[cell_copy.numbers == 3] = 19
        #                 count += 1

        #             #     if i == 0 or i == NSpecies2Change // tmpAtomsDims[1] or i + 1 == NSpecies2Change // tmpAtomsDims[1] or i == tmpAtomsDims[0] - 1:
        #             #         cellCopyScaledCellPar = list(cell_copy.cell.cellpar()[:3])
        #             #         cellCopyScaledCellPar[0] = end1StructureA + end0StructureA
        #             #         cell_copy.set_cell(cellCopyScaledCellPar, scale_atoms=True)
        #             #     elif (i + 1) * tmpAtomsDims[1] <= NSpecies2Change:
        #             #         cellCopyScaledCellPar = list(cell_copy.cell.cellpar()[:3])
        #             #         cellCopyScaledCellPar[0] = 2*end1StructureA
        #             #         cell_copy.set_cell(cellCopyScaledCellPar, scale_atoms=True)
        #             # else:
        #             #     if i == 0 or i == NSpecies2Change // tmpAtomsDims[1] or i + 1 == NSpecies2Change // tmpAtomsDims[1] or i == tmpAtomsDims[0] - 1:
        #             #         cellCopyScaledCellPar = list(cell_copy.cell.cellpar()[:3])
        #             #         cellCopyScaledCellPar[0] = end1StructureA + end0StructureA
        #             #         cell_copy.set_cell(cellCopyScaledCellPar, scale_atoms=True)
                    
        #             print(cell_copy.cell[0], i, NSpecies2Change // tmpAtomsDims[1])
        #             toBeAdded[0] = (cell_copy.cell*[1,1,1])[0]

        #             cell_copy.translate(origin)
        #             origin += cell_copy.cell.dot([0,1,0])
                    
        #             # Append the translated copy to the supercell
        #             tmpAtoms.extend(cell_copy)
                    
        #             # if i == 1 and j == 1:
        #             #     view(tmpAtoms)
        #             #     exit()

        #         supercellCellSize += toBeAdded
        #         origin[1] = 0
        #         origin += cell_copy.cell.dot([1,0,0])
        #         # tmpAtoms.set_cell(supercellCellSize)        
        #     # view(tmpAtoms)
        #     # exit()
        #     tmpAtoms.set_cell(cell_copy.cell)
        #     # view(tmpAtoms)
        #     # exit()

        #     # Wrap the atoms inside the supercell
        #     tmpAtoms.center()
        #     tmpAtomsDims = list(multipicity)
        #     tmpAtomsDims[:2] = [1,1]
        #     tmpAtoms *= tuple(tmpAtomsDims)
        #     tmpAtoms.set_cell(ase_atoms0.cell*multipicity, scale_atoms=False)
        #     tmpAtoms.set_cell(scaledCell[int(NSpecies2Change/multipicity[0] * multipicity[1])]*multipicity, scale_atoms=True)
        #     tmpAtoms.center()    

        ##################################### random #####################################
        tmpAtoms = (ase_atoms0*multipicity).copy()
        
        candi = np.random.choice(np.arange(len(tmpAtoms.numbers))[tmpAtoms.numbers == 3], replace=False, size=int(len(tmpAtoms.numbers)*xxx*0.5))
        tmpAtoms.numbers[candi] = 19
        tmpAtoms.set_cell(scaledCell[int(10000*xxx)], scale_atoms=True)
            
        # BGFS
        chgnet = CHGNet.load()
        SO = StructOptimizer(model=chgnet, optimizer_class="BFGS")
        SO.relax(atoms=AseAtomsAdaptor.get_structure(tmpAtoms), relax_cell=True, save_path=os.path.join(os.getcwd(), str(xxx)), verbose=True)
        
    tmpCharge = tmpAtoms.numbers.copy()
    tmpCharge[tmpCharge == 3] = 1
    tmpCharge[tmpCharge == 17] = -1
    tmpCharge[tmpCharge == 19] = 1
    tmpAtoms.set_initial_charges(tmpCharge)
    write_lammps_data(file=subDir + "/initial-solid.data", atoms=tmpAtoms, atom_style="full")

    symbols = tmpAtoms.get_chemical_symbols()
    species = sorted(set(symbols))
    lmpString = ""
    for s in species:
        # lmpString += " " + str(atomic_numbers[s])
        lmpString += " " + s

    prepare_in = f"""units          metal
atom_style     full

read_data      initial-solid.data
velocity       all create {initialTemp} 1312343 dist gaussian

pair_style     {pair_style}
pair_coeff     {pair_coeff}{lmpString}

variable myZlo equal zlo
variable myZhi equal zhi
variable myMid equal (${{myZlo}}+${{myZhi}})/2

region   solid block INF INF INF INF ${{myZlo}} ${{myMid}}
region   liquid block INF INF INF INF ${{myMid}} ${{myZhi}}
group    gSolid region solid
group    gLiquid region liquid

neighbor        0.3 bin
neigh_modify    delay 10

timestep        0.001
compute msd all msd
thermo_style    custom step temp pe etotal press vol c_msd[4]
thermo          10 
dump            1 all custom 10 xyz-npt.dump type id x y z
#dump           2 all custom 10 force-npt.dump type id fx fy fz

fix             1 all npt temp {initialTemp} {initialTemp} $(100.0*dt) aniso 1.01325 1.01325 $(1000.0*dt)
run             10000
unfix           1

fix             2 gLiquid npt temp {initialTemp} 1500 $(100.0*dt) z 1.01325 1.01325 $(1000.0*dt) couple none
run             20000
unfix           2

write_data      structure.ready.data

write_restart   structure.ready
    """

    with open(subDir + "/initial-npt.in", 'w') as file:
        file.write(prepare_in)    
        
        
#     with open(subDir + "/run-initial-npt.sh", 'w') as f:
#         f.write("""#!/bin/bash
# #SBATCH -J {name}prep # Job name
# #SBATCH -n 8 # Number of total cores
# #SBATCH -N 1 # Number of nodes
# #SBATCH --time=00-02:00:00
# #SBATCH -A cts180021p
# #SBATCH -p RM-shared
# #SBATCH --mem-per-cpu=2000 # Memory pool for all cores in MB
# #SBATCH -e npt.err #change the name of the err file 
# #SBATCH -o npt.out # File to which STDOUT will be written %j is the job #
# #SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
# #SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent


# module load intelmpi/2021.3.0-intel2021.3.0

# conda activate /ocean/projects/cts180021p/wuziqi/conda-envs/CN

# echo "Job started on `hostname` at `date`" 

# mpirun -np $SLURM_NTASKS /ocean/projects/cts180021p/wuziqi/clease/lammps-ASC/build2/lmp -in initial-npt.in > cn.log

# echo " "
# echo "Job Ended at `date`"
# """.format(name="{:.2f}".format(xxx)))
        
        with open(subDir + "/run-initial-npt.sh", 'w') as f:
            f.write("""#!/bin/bash
#SBATCH -J {name}prep # Job name
#SBATCH -n 4 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH --time=00-6:00:00
#SBATCH -p venkvis-a100
#SBATCH --mem=4G # Memory pool for all cores in MB
#SBATCH -e prep.err #change the name of the err file 
#SBATCH -o prep.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=BEGIN,END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent

module load cuda

conda activate /nfs/turbo/coe-venkvis/ziqiw-turbo/conda/t2

echo "Job started on `hostname` at `date`" 

/nfs/turbo/coe-venkvis/ziqiw-turbo/LAMMPSs/lammps-ASC/build/lmp -in initial-npt.in > cn.log

echo " "
echo "Job Ended at `date`"
""".format(name="{:.2f}".format(xxx)))
        
#     with open(subDir + "/initial-melt.in", 'w') as f:
#         f.write("units\tmetal\n")
#         f.write("atom_style\tfull\n")
#         f.write("\n")
#         f.write("read_data\tstructureA.data\n")
#         f.write("\n")
#         # f.write("pair_style\tquip\n")
#         # f.write("pair_coeff\t* * " + potentialPath + " \"Potential xml_label=GAP_2021_9_27_0_1_17_37_618\"" + lmpString + "\n")
#         f.write("pair_style\tchgnet/gpu\t" + chgnetDriverPath + "\n")
#         f.write("pair_coeff\t* *\tMPtrj-efsm" + lmpString + "\n")
#         f.write("\n")
#         f.write("neighbor\t0.3 bin\n")
#         f.write("neigh_modify\tdelay 10\n")
#         f.write("\n")
#         f.write("timestep\t0.001\n")
#         f.write("compute\tmsd all msd\n")
#         f.write("thermo_style\tcustom step temp pe etotal press vol c_msd[4]\n")
#         f.write("thermo\t10\n")
#         f.write("dump\t1 all custom 10 xyz-melt.dump type id x y z\n")
#         f.write("dump\t2 all custom 10 force-melt.dump type id fx fy fz\n")
#         f.write(
#             """
# fix     1 all npt temp 300 2000 $(100.0*dt) z 1.01325 1.01325 $(1000.0*dt) couple none
# run     5000
# unfix   1

# fix     2 all npt temp 2000 2000 $(100.0*dt) z 1.01325 1.01325 $(1000.0*dt) couple none
# run     2000
# unfix   2

# fix     3 all npt temp 2000 300 $(100.0*dt) z 1.01325 1.01325 $(1000.0*dt) couple none
# run     10000
# unfix   3

# fix     4 all npt temp 300 300 $(100.0*dt) z 1.01325 1.01325 $(1000.0*dt) couple none
# run     10000
# unfix   4

# write_data  structureB.data
#             """
#         )
    
#         with open(subDir + "/run-initial-melt.sh", 'w') as f:
#             f.write("""#!/bin/bash
# #SBATCH -J {name}melt # Job name
# #SBATCH -n 16 # Number of total cores
# #SBATCH -N 1 # Number of nodes
# #SBATCH --time=00-08:00:00
# #SBATCH -A cts180021p
# #SBATCH -p RM-shared
# #SBATCH --mem-per-cpu=2000 # Memory pool for all cores in MB
# #SBATCH -e melt.err #change the name of the err file 
# #SBATCH -o melt.out # File to which STDOUT will be written %j is the job #
# #SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
# #SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent


# module load intelmpi/2021.3.0-intel2021.3.0

# echo "Job started on `hostname` at `date`" 

# mpirun -np $SLURM_NTASKS /ocean/projects/cts180021p/wuziqi/clease/lammps-stable_29Sep2021_update3/GAPbuild/lmp -in initial-melt.in > melt.log

# echo " "
# echo "Job Ended at `date`"
#             """.format(name="{:.2f}".format(x)))
        
    with open(subDir + "/123toABC.txt", 'w') as f:
        f.write(lmpString)

print("Step0 done!")

