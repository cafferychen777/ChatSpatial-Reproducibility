#!/bin/bash
#SBATCH -J oscc157_env
#SBATCH -p medium
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
# Env for the full OSCC case-study re-run through ChatSpatial 1.5.7 MCP tools.
# Clone of chatspatial_bx156 (Python 3.12 + R + rpy2; CellChat in $S/expB/no_user_rlib), then chatspatial[full] 1.5.7:
# PyPI if released, otherwise the GitHub release commit 3d21a74 (main, "Release ChatSpatial 1.5.7").
set -uo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
W=$S/oscc157
eval "$(conda shell.bash hook)"
export TMPDIR=$W/tmp PYTHONNOUSERSITE=1 PIP_CACHE_DIR=$S/PIP_CACHE
E=$S/conda_envs/chatspatial_v157
[ -x $E/bin/python ] || conda create -y -p $E --clone $S/conda_envs/chatspatial_bx156 >/dev/null || { echo CLONE_FAILED; exit 2; }
conda activate $E
if python -m pip download --no-deps -d $W/tmp/pypi_probe "chatspatial==1.5.7" >/dev/null 2>&1; then
  echo "installing chatspatial[full]==1.5.7 from PyPI"
  python -m pip install "chatspatial[full]==1.5.7" 2>&1 | tail -3
else
  echo "1.5.7 not on PyPI yet; installing GitHub commit 3d21a74"
  python -m pip install "chatspatial[full] @ https://github.com/cafferychen777/ChatSpatial/archive/3d21a74fff4957056a0fba41476ea8341c523f5d.tar.gz" 2>&1 | tail -3
fi
export R_LIBS_USER=$S/expB/no_user_rlib R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
python - <<'PY'
import importlib.metadata as md
for p in ["chatspatial", "flashdeconv", "mcp", "scanpy", "squidpy", "anndata", "numpy", "scipy", "pandas", "statsmodels", "rpy2", "leidenalg", "igraph", "torch"]:
    try:
        print(p, md.version(p))
    except Exception as e:
        print(p, "MISSING")
import chatspatial
print("import chatspatial", chatspatial.__version__)
PY
Rscript -e 'suppressMessages(library(CellChat)); cat("R", R.version.string, "CellChat", as.character(packageVersion("CellChat")), "\n")'
python -m pip freeze > $W/env_freeze_chatspatial_v157.txt
echo BUILD_DONE
