#!/bin/bash
#SBATCH --job-name=expA_dlpfc
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prep_dlpfc_%j.log
# Build all 12 DLPFC sections on scratch (raw counts + benchmark-preprocessed + labels).
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
ROOT=$S/chatspatial_expA
module load R/4.5.2-src-2026a
eval "$(conda shell.bash hook)"
conda activate ${EXPA_ENV:-$S/conda_envs/chatspatial_expA_155}
python "$ROOT/scripts/dlpfc_prepare_data.py" --out-dir "$ROOT/data/dlpfc" "$@"
du -sh "$ROOT"/data/dlpfc/*
