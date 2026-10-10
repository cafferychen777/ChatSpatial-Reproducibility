#!/bin/bash
# Experiment A on the released ChatSpatial 1.5.5 (cluster login node; writes task
# files via srun and submits jobs). Result tags: dlpfc155, noise155, platforms155.
# Concurrency caps as in the pilot: 60 light (4 CPU) + 24 deep (16 CPU) tasks;
# while the noise and platforms stages overlap, the caps are split between them.
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
ROOT=$S/chatspatial_expA
export EXPA_ENV=$S/conda_envs/chatspatial_expA_155
export EXPA_CACHE=$ROOT/cache/train155
export EXPA_TIMEOUT=3600   # runtime cap for every backend (package default: 3600 STAGATE/GraphST, 600 others)
eval "$(conda shell.bash hook)"
conda activate "$EXPA_ENV"
cd "$ROOT/scripts"
tasks() { if [ -n "${SLURM_JOB_ID:-}" ]; then python make_tasks.py "$@"; else srun -n 1 -p short -c 1 --mem=4G -t 15 python make_tasks.py "$@"; fi; }
OUT=$(tasks dlpfc --name dlpfc155); echo "$OUT"
N=$(echo "$OUT" | sed -n 's/^dlpfc155: \([0-9]*\) tasks.*/\1/p' | head -1); L=$(echo "$OUT" | sed -n 's/^light \([0-9]*\)$/\1/p' | head -1)
J1=$(sbatch --parsable --export=ALL --array=0-$((L-1))%60 -c 4 --mem=24G --time=08:00:00 --job-name=expA_dlpfc155 run_array.sbatch dlpfc155)
J2=$(sbatch --parsable --export=ALL --array=${L}-$((N-1))%24 -c 16 --mem=48G --time=1-00:00:00 --job-name=expA_dlpfc155_deep run_array.sbatch dlpfc155)
J3=$(NOISE_PAR=40 NOISE_PAR_DEEP=16 sbatch --parsable --export=ALL,NOISE_PAR=40,NOISE_PAR_DEEP=16 --dependency=afterany:$J1:$J2 noise_stage.sbatch dlpfc155 noise155)
OUT=$(tasks platforms --name platforms155); echo "$OUT"
NP=$(echo "$OUT" | sed -n 's/^platforms155: \([0-9]*\) tasks.*/\1/p' | head -1); LP=$(echo "$OUT" | sed -n 's/^light \([0-9]*\)$/\1/p' | head -1)
J4=$(sbatch --parsable --export=ALL --dependency=afterany:$J1:$J2 --array=0-$((LP-1))%20 -c 4 --mem=48G --time=1-00:00:00 --job-name=expA_platforms155 run_array.sbatch platforms155)
J5=$(sbatch --parsable --export=ALL --dependency=afterany:$J1:$J2 --array=${LP}-$((NP-1))%8 -c 16 --mem=64G --time=1-00:00:00 --job-name=expA_platforms155_deep run_array.sbatch platforms155)
echo "dlpfc155: light $J1, deep $J2 | select+noise155: $J3 | platforms155: light $J4, deep $J5"
