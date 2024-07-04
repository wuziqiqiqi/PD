#!/bin/bash
#SBATCH -J 3.00prep # Job name
#SBATCH -n 4 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH --time=00-4:00:00
#SBATCH -A cts180021p
#SBATCH -p GPU-shared
#SBATCH --mem=4096 # Memory pool for all cores in MB
#SBATCH -e prep.err #change the name of the err file 
#SBATCH -o prep.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=BEGIN,END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent

module load anaconda3
module load intelmpi/2021.3.0-intel2021.3.0

conda activate /ocean/projects/cts180021p/wuziqi/conda-envs/CN

echo "Job started on `hostname` at `date`" 

/ocean/projects/cts180021p/wuziqi/clease/lammps-ASC/build2/lmp -in initial-npt.in > cn.log

echo " "
echo "Job Ended at `date`"