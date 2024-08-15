#!/bin/bash
#SBATCH -J emc # Job name
#SBATCH -n 4 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --time=00-4:00:00
#SBATCH -p venkvis-cpu
#SBATCH --mem=8192 # Memory pool for all cores in MB
#SBATCH -e emc.err #change the name of the err file 
#SBATCH -o emc.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=BEGIN,END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent

source /nfs/turbo/coe-venkvis/ziqiw-turbo/.bashrc
conda activate /nfs/turbo/coe-venkvis/ziqiw-turbo/conda/casm
cd /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD

echo "Job started on `hostname` at `date`" 

python /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD/fmc.py

echo " "
echo "Job Ended at `date`"
