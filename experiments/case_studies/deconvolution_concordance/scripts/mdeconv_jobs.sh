#!/bin/bash
# M-DECONV: submit prep -> 12-sample array -> analysis on the cluster (run from a login node;
# it only calls sbatch). Inputs are staged in $W/input beforehand.
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
W=$S/mdeconv
mkdir -p $W/logs $W/ref $W/runs $W/analysis $W/tmp
ENV_LINES="eval "$(conda shell.bash hook)"
conda activate $S/conda_envs/chatspatial_v155
export R_LIBS_USER=$S/expB/rlib_v155:$S/expB/no_user_rlib R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg TMPDIR=$W/tmp NUMBA_CACHE_DIR=$W/tmp/numba
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4"

P=$(sbatch --parsable -J mdeconv_prep -p short -c 2 --mem=24G -t 01:00:00 -o $W/logs/%x_%j.out --wrap "
$ENV_LINES
python -W ignore $W/scripts/mdeconv_prep.py $W/input/GSE103322_HNSCC_all_data.txt.gz $W/ref")
A=$(sbatch --parsable -J mdeconv_run -p medium -c 4 --mem=40G -t 08:00:00 --array=1-12 -d afterok:$P \
  -o $W/logs/%x_%A_%a.out --wrap "
$ENV_LINES
python -W ignore $W/scripts/mdeconv_run.py s\${SLURM_ARRAY_TASK_ID} $W/input/processed_h5ad $W/ref $W/runs")
Z=$(sbatch --parsable -J mdeconv_analyze -p short -c 1 --mem=8G -t 00:30:00 -d afterany:$A \
  -o $W/logs/%x_%j.out --wrap "
$ENV_LINES
python -W ignore $W/scripts/mdeconv_analyze.py $W/runs $W/analysis")
echo "prep=$P array=$A analyze=$Z"
