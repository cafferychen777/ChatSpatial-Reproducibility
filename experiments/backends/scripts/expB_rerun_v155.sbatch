#!/bin/bash
#SBATCH -J expB_v155
#SBATCH -p medium
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o expB_logs/%x_%j.out
# Re-execute every unique recorded call of experiment B (no LLM calls) under ChatSpatial 1.5.5, then rerun the analysis.
set -uo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $S/conda_envs/chatspatial_v155
export R_LIBS_USER=$S/expB/rlib_v155:$S/expB/no_user_rlib R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg EXPB_DIR=$S/expB EXPB_FIXTURES=$S/expB/fixtures \
       EXPB_EXEC_DIR=exec_v155 EXPB_ANALYSIS_DIR=analysis_v155 TMPDIR=$S/expB/tmp \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
cd ${EXPB_SCRIPTS:-$SLURM_SUBMIT_DIR}   # submit from experiments/backends/scripts
echo "$(date -Is) host=$(hostname) chatspatial=$(python -c 'import chatspatial;print(chatspatial.__version__)')"
python -W ignore expB_execute.py --files "$S/expB/invocation/*.jsonl" --jobs 2 --timeout 2700
echo "$(date -Is) EXEC_DONE"
python -W ignore expB_analysis.py
echo "$(date -Is) ANALYSIS_DONE"
