#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=gradientboosting
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yt1209@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python/intel/2.7.12
module load scikit-learn/intel/0.18.1

python gradientboosting_python.py fourmillion_withItemID_unwrapped.csv

# leave a blank line at the end