#!/bin/bash
#SBATCH -J emc1 # Job name
#SBATCH -n 2 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --time=00-8:00:00
#SBATCH -p venkvis-h100,venkvis-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=4000 # Memory pool for all cores in MB
#SBATCH -e emc1.err #change the name of the err file 
#SBATCH -o emc1.out # File to which STDOUT will be written %j is the job #

source /nfs/turbo/coe-venkvis/ziqiw-turbo/.bashrc
conda activate /nfs/turbo/coe-venkvis/ziqiw-turbo/conda/casm
cd /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD

echo "Job started on `hostname` at `date`" 

python /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD/fmc.py --input="LiMg-example-0.8-1000relax.yaml" -b=false -d=gpu --gs=[1]

echo " "
echo "Job Ended at `date`"
