#!/bin/bash
#SBATCH -J 0.80-244.0prep # Job name
#SBATCH -n 4 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH --time=00-6:00:00
#SBATCH -p venkvis-a100
#SBATCH --mem=4096 # Memory pool for all cores in MB
#SBATCH -e prep.err #change the name of the err file 
#SBATCH -o prep.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=BEGIN,END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent

module load cuda

conda activate /nfs/turbo/coe-venkvis/ziqiw-turbo/conda/t2

echo "Job started on `hostname` at `date`" 

/nfs/turbo/coe-venkvis/ziqiw-turbo/LAMMPSs/lammps-ASC/build/lmp -in mixed.in > mixed.log

echo " "
echo "Job Ended at `date`"
        