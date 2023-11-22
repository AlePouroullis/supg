#!/bin/sh
#SBATCH --account=compsci
#SBATCH --partition=curie
#SBATCH --time=180:00:00
#SBATCH --nodes=1 --ntasks=32
#SBATCH --job-name="ECSUPGJob12"
#SBATCH --mail-user=smtjul022@uct.ac.za
#SBATCH --mail-type=ALL
module load python/anaconda-python-3.7
 
python3 /home/smtjul022/JSUPG/example_run.py 10000 ECSUPG12