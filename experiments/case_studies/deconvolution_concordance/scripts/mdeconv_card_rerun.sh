#!/bin/bash
# M-DECONV: rerun CARD only (first attempt failed: library(MuSiC) missing in chatspatial_v155),
# with MuSiC from $W/rlib_music, then rerun the analysis after the main array and the CARD array finish.
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}; W=$S/mdeconv
MAIN=${1:?main array job id}
ENV_LINES="eval "$(conda shell.bash hook)"
conda activate $S/conda_envs/chatspatial_v155
export R_LIBS_USER=$W/rlib_music:$S/expB/rlib_v155:$S/expB/no_user_rlib R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg TMPDIR=$W/tmp NUMBA_CACHE_DIR=$W/tmp/numba
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4"
mkdir -p $W/runs_card
C=$(sbatch --parsable -J mdeconv_card -p medium -c 4 --mem=40G -t 06:00:00 --array=1-12 \
  -o $W/logs/%x_%A_%a.out --wrap "
$ENV_LINES
python -W ignore $W/scripts/mdeconv_run.py s\${SLURM_ARRAY_TASK_ID} $W/input/processed_h5ad $W/ref $W/runs_card card")
Z=$(sbatch --parsable -J mdeconv_analyze2 -p short -c 1 --mem=8G -t 00:30:00 -d afterany:$MAIN:$C \
  -o $W/logs/%x_%j.out --wrap "
$ENV_LINES
mkdir -p $W/runs_final
cp $W/runs/*.csv $W/runs_final/ 2>/dev/null || true
cp $W/runs_card/*__card.csv $W/runs_final/ 2>/dev/null || true
python - <<'PY'
import json,glob
W='$W'
for s in range(1,13):
    a=json.load(open(f'{W}/runs/s{s}__runlog.json')) if glob.glob(f'{W}/runs/s{s}__runlog.json') else []
    b=json.load(open(f'{W}/runs_card/s{s}__runlog.json')) if glob.glob(f'{W}/runs_card/s{s}__runlog.json') else []
    a=[r for r in a if r['run']!='card']+b
    json.dump(a,open(f'{W}/runs_final/s{s}__runlog.json','w'),indent=1)
PY
python -W ignore $W/scripts/mdeconv_analyze.py $W/runs_final $W/analysis")
echo "card=$C analyze2=$Z"
