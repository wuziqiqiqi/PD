#!/bin/bash
#SBATCH -J 0.20-975.0 # Job name
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
                