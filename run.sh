#!/bin/bash
#SBATCH -J emc # Job name
#SBATCH -n 4 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH --time=00-8:00:00
#SBATCH -p venkvis-h100
#SBATCH --mem=4096 # Memory pool for all cores in MB
#SBATCH -e emc.err #change the name of the err file 
#SBATCH -o emc.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=BEGIN,END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent

source /home/ziqiw/.bashrc
conda activate /nfs/turbo/coe-venkvis/ziqiw-turbo/conda/casm
cd /nfs/turbo/coe-venkvis/ziqiw-turbo/PD/solid/clease

echo "Job started on `hostname` at `date`" 

python /nfs/turbo/coe-venkvis/ziqiw-turbo/PD/solid/clease/emc.py

echo " "
echo "Job Ended at `date`"
