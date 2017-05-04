
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=xgboost
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yt1209@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.5.3
module load scikit-learn/intel/0.18.1



DATADIR=$SCRATCH/yt1209/MLproject_data/xgboost
python -W ignore xgboost_train.py $DATADIR/sample1.csv