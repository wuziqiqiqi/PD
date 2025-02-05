#!/bin/bash
#SBATCH -J emc1 # Job name
#SBATCH -A bcuf-delta-gpu # Account to charge
#SBATCH -n 2 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --time=8:00:00
#SBATCH -p gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --mem=4000 # Memory pool for all cores in MB
#SBATCH -e emc1.err #change the name of the err file 
#SBATCH -o emc1.out # File to which STDOUT will be written %j is the job #

source /u/wuziqi/.bashrc
conda activate casm
cd /u/wuziqi/PD

echo "Job started on `hostname` at `date`" 

python /u/wuziqi/PD/fmcAnySpecies.py --input="AgPt-example-0.8-1000relax-mattersim.yaml" --device=gpu --batch=false --gs=[0]

echo " "
echo "Job Ended at `date`"
