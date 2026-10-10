#!/bin/bash
#SBATCH -J bx_run
#SBATCH -p short
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 01:55:00
#SBATCH -o logs/%x_%A_%a.out
# One experiment B-X trial per array element. Usage:
#   sbatch --array=0-N%K expBX_array.sbatch TASKFILE [OFFSET]
set -uo pipefail
module load Python/3.11.5-GCCcore-13.2.0 >/dev/null 2>&1 || true
IDX=$(( ${SLURM_ARRAY_TASK_ID} + ${2:-0} ))
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}
python3 $S/expBX/scripts/expBX_run_task.py "$1" "$IDX"
