#!/bin/bash
#SBATCH -J 40.00prep # Job name
#SBATCH -n 8 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --time=00-02:00:00
#SBATCH -A cts180021p
#SBATCH -p RM-shared
#SBATCH --mem-per-cpu=2000 # Memory pool for all cores in MB
#SBATCH -e npt.err #change the name of the err file 
#SBATCH -o npt.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent

module load anaconda3

module load intelmpi/2021.3.0-intel2021.3.0

conda activate /ocean/projects/cts180021p/wuziqi/conda-envs/CN

echo "Job started on `hostname` at `date`" 

mpirun -np $SLURM_NTASKS /ocean/projects/cts180021p/wuziqi/clease/lammps-ASC/build2/lmp -in initial-npt.in > cn.log

echo " "
echo "Job Ended at `date`"
