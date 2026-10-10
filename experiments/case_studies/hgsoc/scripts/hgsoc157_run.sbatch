#!/bin/bash
#SBATCH -J hgsoc157_run
#SBATCH -p medium
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
# HGSOC case study end to end through ChatSpatial 1.5.7 MCP tools, then table collection and statistics.
# HGSOC_SAMPLES / HGSOC_OUT override the sections and the output directory (smoke test).
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
W=$S/hgsoc157
eval "$(conda shell.bash hook)"
conda activate $S/conda_envs/chatspatial_v157
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export R_LIBS_USER=$S/expB/no_user_rlib R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
# Intermediates on node-local disk (scratch file-count quota); call record, tables and logs are copied to $OUT.
L=/tmp/hgsoc157_${SLURM_JOB_ID}
OUT=${HGSOC_OUT:-$W/out}
mkdir -p $L/runs $L/tmp/numba $L/tmp/cs_cache $OUT
export TMPDIR=$L/tmp NUMBA_CACHE_DIR=$L/tmp/numba HGSOC_RUN_DIR=$L/runs
export CHATSPATIAL_OUTPUT_DIR=$L/runs/chatspatial_output CHATSPATIAL_CACHE_DIR=$L/tmp/cs_cache HOME_CS=$L
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMBA_NUM_THREADS=16
python -c "import chatspatial, infercnvpy, sys; print('python', sys.version.split()[0], 'chatspatial', chatspatial.__version__, 'infercnvpy', infercnvpy.__version__)"
python $W/scripts/hgsoc157_mcp_driver.py $W 2> $L/server_stderr.log || { cp $L/server_stderr.log $L/runs/hgsoc157_tool_calls.jsonl $OUT/ ; tail -80 $L/server_stderr.log; rm -rf $L; exit 1; }
python -W ignore $W/scripts/hgsoc157_collect.py $L/runs $L/tables
python -W ignore $W/scripts/hgsoc157_stats.py $L/tables $L/tables/stats P3 || echo STATS_FAILED
rsync -a $L/runs/hgsoc157_tool_calls.jsonl $L/tables $OUT/
cp $L/server_stderr.log $OUT/server_stderr_${SLURM_JOB_ID}.log
rm -rf $L
echo RUN_DONE
