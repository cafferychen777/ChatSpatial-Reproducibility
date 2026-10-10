#!/bin/bash
#SBATCH -J bx_v156env
#SBATCH -p medium
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
# ChatSpatial 1.5.6 env for the B-X ChatSpatial arms: clone of chatspatial_v155 (the experiment-B 1.5.5 env, R 4.4.3,
# rpy2 3.5.17) + chatspatial[full]==1.5.6 from PyPI (no --upgrade: only what 1.5.6 requires changes). Freeze diff logged.
set -uo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}
source ~/miniconda3/etc/profile.d/conda.sh
export TMPDIR=$S/expBX/tmp PYTHONNOUSERSITE=1
E=$S/conda_envs/chatspatial_bx156
[ -x $E/bin/python ] || conda create -y -p $E --clone $S/conda_envs/chatspatial_v155 >/dev/null || { echo CLONE_FAILED; exit 2; }
conda activate $E
python -m pip freeze > $S/expBX/logs/v156_freeze_before.txt
python -m pip install "chatspatial[full]==1.5.6" --no-cache-dir 2>&1 | tail -3
python -c "import numpy; print('numpy', numpy.__version__)"
NP=$($S/conda_envs/chatspatial_v155/bin/python -c "import numpy;print(numpy.__version__)")
python -c "import numpy,sys; sys.exit(0 if numpy.__version__=='$NP' else 1)" || python -m pip install --force-reinstall --no-deps "numpy==$NP" 2>&1 | tail -1
python -m pip freeze > $S/expBX/logs/v156_freeze_after.txt
echo "== freeze diff vs v155 clone"; diff $S/expBX/logs/v156_freeze_before.txt $S/expBX/logs/v156_freeze_after.txt
export R_LIBS_USER=$S/expB/rlib_v155:$S/expB/no_user_rlib R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
python -c "import chatspatial, rpy2, mcp; print('chatspatial', chatspatial.__version__, 'rpy2', rpy2.__version__)"
python -m pip show mcp | grep Version
python -c "import rpy2.robjects as ro; ro.r('suppressMessages({library(spacexr); library(CellChat); library(SPARK); library(CARD)})'); print('R pkgs OK')" 2>&1 | tail -1
for m in STAGATE_pyG GraphST SpaGCN cellrank infercnvpy liana SpatialDE flashdeconv fastccc; do python -W ignore -c "import $m" >/dev/null 2>&1 && echo "PY $m ok" || echo "PY $m MISSING"; done
echo BUILD_DONE
