#!/bin/bash
#SBATCH -J oscc157_run
#SBATCH -p medium
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
# OSCC case study end to end through ChatSpatial 1.5.7 MCP tools, then table collection.
# OSCC_SAMPLES / OSCC_RUN_DIR override the sections and run directory (smoke test).
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
W=$S/oscc157
eval "$(conda shell.bash hook)"
conda activate $S/conda_envs/chatspatial_v157
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export R_LIBS_USER=$S/expB/no_user_rlib R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
# Intermediates on node-local disk (the scratch file-count quota was full on 2026-10-09); the call record,
# tables and final h5ad files are copied to $OUT afterwards.
L=/tmp/oscc157_${SLURM_JOB_ID}
OUT=${OSCC_OUT:-$W/out}
mkdir -p $L/runs $L/tmp/numba $L/tmp/cs_cache $OUT
export TMPDIR=$L/tmp NUMBA_CACHE_DIR=$L/tmp/numba OSCC_RUN_DIR=$L/runs
export CHATSPATIAL_OUTPUT_DIR=$L/runs/chatspatial_output CHATSPATIAL_CACHE_DIR=$L/tmp/cs_cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
python -c "import chatspatial, flashdeconv, sys; print('python', sys.version.split()[0], 'chatspatial', chatspatial.__version__, 'flashdeconv', flashdeconv.__version__)"
python $W/scripts/oscc157_mcp_driver.py $W 2> $L/server_stderr.log || { cp $L/server_stderr.log $L/runs/oscc157_tool_calls.jsonl $OUT/ ; tail -50 $L/server_stderr.log; exit 1; }
python -W ignore $W/scripts/oscc157_collect.py $W $L/runs/tables
rsync -a $L/runs/oscc157_tool_calls.jsonl $L/runs/tables $L/runs/final_h5ad $L/runs/cellchat $OUT/
cp $L/server_stderr.log $OUT/server_stderr_${SLURM_JOB_ID}.log
rm -rf $L
echo RUN_DONE
