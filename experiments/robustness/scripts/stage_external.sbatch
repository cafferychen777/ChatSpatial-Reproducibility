#!/bin/bash
#SBATCH --job-name=expA_stage
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stage_external_%j.log
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
ROOT=$S/chatspatial_expA
eval "$(conda shell.bash hook)"
conda activate ${EXPA_ENV:-$S/conda_envs/chatspatial_expA_155}
python "$ROOT/scripts/stage_external.py" "$@"
du -sh "$ROOT"/data/*
