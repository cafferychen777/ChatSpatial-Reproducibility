#!/bin/bash
#SBATCH -J hgsoc157_ref
#SBATCH -p medium
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
eval "$(conda shell.bash hook)"
conda activate $S/conda_envs/chatspatial_v157
export PYTHONNOUSERSITE=1
python -W ignore $S/hgsoc157/scripts/hgsoc157_prep_reference.py $S/hgsoc157
