#!/bin/bash
#SBATCH -J emc # Job name
#SBATCH -n 1 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --time=00-8:00:00
#SBATCH -p venkvis-h100
#SBATCH --gres=gpu:1
#SBATCH --mem=2000 # Memory pool for all cores in MB
#SBATCH -e emc-chg.err #change the name of the err file 
#SBATCH -o emc-chg.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=BEGIN,END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent

source /nfs/turbo/coe-venkvis/ziqiw-turbo/.bashrc
conda activate /nfs/turbo/coe-venkvis/ziqiw-turbo/conda/casm
cd /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD

echo "Job started on `hostname` at `date`" 

python /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD/fmc.py --input="LiMg-example-0.8-5000relax.yaml" -b=false -d=gpu

echo " "
echo "Job Ended at `date`"
