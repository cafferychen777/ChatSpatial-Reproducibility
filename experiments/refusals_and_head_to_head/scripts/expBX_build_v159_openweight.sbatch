#!/bin/bash
#SBATCH -J bx_v159env
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
# aarch64 ChatSpatial 1.5.9 env for the Qwen head-to-head arm on the GPU host: same recipe as cluster job 2725920
# (clone of chatspatial_bx156 + chatspatial[full]==1.5.9, no --upgrade, numpy kept). The clone is a byte copy with the
# equal-length prefix chatspatial_bx156 -> chatspatial_bx159 replaced in every file (what conda clone does for prefixes).
set -uo pipefail
B=$HOME/expBX
O=$B/envs/chatspatial_bx156
E=$B/envs/chatspatial_bx159
export TMPDIR=$B/tmp PYTHONNOUSERSITE=1
if [ ! -x $E/bin/python ]; then
  cp -a $O $E || { echo COPY_FAILED; exit 2; }
  grep -rlZ --binary-files=binary 'chatspatial_bx156' $E 2>/dev/null | xargs -0 -r perl -pi -e 's/chatspatial_bx156/chatspatial_bx159/g'
  echo "remaining old-prefix files: $(grep -rl 'chatspatial_bx156' $E 2>/dev/null | wc -l)"
fi
PY=$E/bin/python
$PY -m pip freeze > $B/logs/v159_freeze_before.txt
$PY -m pip install "chatspatial[full]==1.5.9" --no-cache-dir 2>&1 | tail -3
NP=$($O/bin/python -c "import numpy;print(numpy.__version__)")
$PY -c "import numpy,sys; sys.exit(0 if numpy.__version__=='$NP' else 1)" || $PY -m pip install --force-reinstall --no-deps "numpy==$NP" 2>&1 | tail -1
$PY -m pip freeze > $B/logs/v159_freeze_after.txt
echo "== freeze diff vs bx156 copy"; diff $B/logs/v159_freeze_before.txt $B/logs/v159_freeze_after.txt
export PATH=$E/bin:/usr/bin:/bin R_LIBS_USER= R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
echo "R HOME: $($E/bin/R RHOME)"
$PY -W ignore -c "import chatspatial, rpy2, mcp; print('chatspatial', chatspatial.__version__, 'rpy2', rpy2.__version__)"
$PY -m pip show mcp | grep Version
$PY -c "import rpy2.robjects as ro; print(ro.r('R.home()')[0]); ro.r('suppressMessages({library(spacexr); library(CellChat); library(SPARK); library(CARD)})'); print('R pkgs OK')" 2>&1 | tail -2
for m in STAGATE_pyG GraphST SpaGCN cellrank infercnvpy liana SpatialDE flashdeconv fastccc torch; do $PY -W ignore -c "import $m" >/dev/null 2>&1 && echo "PY $m ok" || echo "PY $m MISSING"; done
echo BUILD_DONE
