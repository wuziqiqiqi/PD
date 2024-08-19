#!/bin/bash
#SBATCH -J emc # Job name
#SBATCH -n 1 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --time=00-8:00:00
#SBATCH -p venkvis-cpu
#SBATCH --mem=2000 # Memory pool for all cores in MB
#SBATCH -e outNerr/emc.err #change the name of the err file 
#SBATCH -o outNerr/emc.out # File to which STDOUT will be written %j is the job #

source /nfs/turbo/coe-venkvis/ziqiw-turbo/.bashrc
conda activate /nfs/turbo/coe-venkvis/ziqiw-turbo/conda/casm
cd /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD


echo "Job started on `hostname` at `date`" 

python fmc.py --input="LiMg-example-0.6-5000relax.yaml" --batch=true --init=true

echo " "
echo "Job Ended at `date`"
