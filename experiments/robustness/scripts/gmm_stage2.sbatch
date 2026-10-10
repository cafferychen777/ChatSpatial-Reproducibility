#!/bin/bash
#SBATCH --job-name=expA_gmm_stage2
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# After the GMM-fix re-runs of dlpfc155: re-select with the frozen rule, rebuild the
# noise155 task list, purge affected noise units and re-run every incomplete noise task.
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
ROOT=$S/chatspatial_expA
eval "$(conda shell.bash hook)"
conda activate "$EXPA_ENV"
cd "$ROOT/scripts"
export PYTHONWARNINGS=ignore PYTHONPATH=$EXPA_PYTHONPATH_PREFIX:$ROOT/scripts
python incomplete_tasks.py dlpfc155
python select_rule.py dlpfc155
python make_tasks.py noise --from-stage dlpfc155 --name noise155
python purge_affected.py noise155 --apply
IDX=$(python incomplete_tasks.py noise155)
echo "noise155 tasks to run: $IDX"
[ -n "$IDX" ] && sbatch --parsable --export=ALL --array=${IDX}%24 -c 8 --mem=32G --time=1-00:00:00 \
    --job-name=expA_noise155_gmm run_array.sbatch noise155
