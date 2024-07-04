#!/bin/bash
#SBATCH -J 0.30prep # Job name
#SBATCH -n 4 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH --time=00-6:00:00
#SBATCH -p venkvis-h100
#SBATCH --mem=4096 # Memory pool for all cores in MB
#SBATCH -e prep.err #change the name of the err file 
#SBATCH -o prep.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=BEGIN,END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ziqiw@umich.edu # Email to which notifications will be sent

module load spack/0.21
unset SPACK_DISABLE_LOCAL_CONFIG
module load anaconda3

source /nfs/turbo/coe-venkvis/ziqiw-turbo/.bashrc

conda activate chgnet

echo "Job started on `hostname` at `date`" 

/nfs/turbo/coe-venkvis/ziqiw-turbo/LAMMPSs/lammps-ASC/build/lmp -in initial-npt.in > cn.log

echo " "
echo "Job Ended at `date`"
