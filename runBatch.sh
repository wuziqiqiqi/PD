#!/bin/bash
#SBATCH --array=0-20   # Array index range
#SBATCH -J emc # Job name
#SBATCH -n 1 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --time=00-8:00:00
#SBATCH -p venkvis-cpu
#SBATCH --mem=2000 # Memory pool for all cores in MB
#SBATCH -e outNerr/emc_%A_%a.err #change the name of the err file 
#SBATCH -o outNerr/emc_%A_%a.out # File to which STDOUT will be written %j is the job #

source /nfs/turbo/coe-venkvis/ziqiw-turbo/.bashrc
conda activate /nfs/turbo/coe-venkvis/ziqiw-turbo/conda/casm
cd /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD

# Define the array of custom values
VALUES=(100.0 135.0 170.0 205.0 240.0 275.0 310.0 345.0 380.0 415.0 450.0 485.0 520.0 555.0 590.0 625.0 660.0 695.0 730.0 765.0 800.0)

# Get the value corresponding to the current task ID
MY_VALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Running task $SLURM_ARRAY_TASK_ID with value $MY_VALUE"

# Example: Run a Python script with the selected value
python fmc.py --input=LiMg-example-0.8-5000relax-3.yaml --batch=true --init=false --temp=$MY_VALUE
