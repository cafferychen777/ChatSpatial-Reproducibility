#!/bin/bash
#SBATCH --job-name=expA_select
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# Usage: sbatch noise_stage.sbatch MAIN_STAGE NOISE_NAME   (defaults: pilot noise)
# Apply the frozen label-free rule to MAIN_STAGE, then submit the noise arrays.
set -euo pipefail
MAIN=${1:-pilot}; NOISE=${2:-noise}
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
ROOT=$S/chatspatial_expA
eval "$(conda shell.bash hook)"
conda activate ${EXPA_ENV:-$S/conda_envs/chatspatial_expA_155}
cd "$ROOT/scripts"
export PYTHONWARNINGS=ignore
python select_rule.py "$MAIN"
OUT=$(python make_tasks.py noise --from-stage "$MAIN" --name "$NOISE"); echo "$OUT"
N=$(echo "$OUT" | sed -n "s/^$NOISE: \([0-9]*\) tasks.*/\1/p" | head -1)
L=$(echo "$OUT" | sed -n 's/^light \([0-9]*\)$/\1/p' | head -1)
sbatch --parsable --export=ALL --array=0-$((L-1))%${NOISE_PAR:-60} -c 4 --mem=24G --time=08:00:00 \
       --job-name=expA_${NOISE} run_array.sbatch "$NOISE"
sbatch --parsable --export=ALL --array=${L}-$((N-1))%${NOISE_PAR_DEEP:-24} -c 16 --mem=48G --time=1-00:00:00 \
       --job-name=expA_${NOISE}_deep run_array.sbatch "$NOISE"
