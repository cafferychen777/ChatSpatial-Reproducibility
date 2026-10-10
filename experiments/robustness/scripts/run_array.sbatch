#!/bin/bash
#SBATCH --job-name=expA
#SBATCH --partition=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.log
# Usage: sbatch --array=0-N%P --job-name=expA_<stage> run_array.sbatch <stage>
# Env vars (pass with --export=ALL,...): EXPA_ENV (conda env), EXPA_CACHE (training cache dir),
# EXPA_TIMEOUT (runtime cap, s), EXPA_STRIP_NEIGHBORS / EXPA_GRAPHST_RESEED /
# EXPA_MANUAL_STAGATE_RADIUS = 1 reproduce the pre-release pilot workarounds.
set -uo pipefail
STAGE=$1
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
ROOT=$S/chatspatial_expA
eval "$(conda shell.bash hook)"
conda activate ${EXPA_ENV:-$S/conda_envs/chatspatial_expA_155}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CHATSPATIAL_EXPORT_RESULTS=0 PYTHONWARNINGS=ignore MPLBACKEND=Agg
export EXPA_TRAIN_CACHE=${EXPA_CACHE:-$ROOT/cache/train}
export PYTHONPATH=${EXPA_PYTHONPATH_PREFIX:+$EXPA_PYTHONPATH_PREFIX:}$ROOT/scripts/cache_hook:$ROOT/scripts
python "$ROOT/scripts/run_task.py" --tasks-dir "$ROOT/tasks/$STAGE"
