#!/bin/bash
#SBATCH -J hgsoc157_prep
#SBATCH -p medium
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
W=$S/hgsoc157
eval "$(conda shell.bash hook)"
conda activate $S/conda_envs/chatspatial_v157
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export R_LIBS_USER=$S/expB/no_user_rlib R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
python -W ignore $W/scripts/hgsoc157_prep_inputs.py $W
python - <<'PY'
import importlib.metadata as md
for p in ["chatspatial", "infercnvpy", "scanpy", "squidpy", "anndata", "rpy2", "mcp", "leidenalg"]:
    try: print(p, md.version(p))
    except Exception: print(p, "MISSING")
PY
Rscript -e 'suppressMessages(library(spacexr)); cat("R", R.version.string, "spacexr", as.character(packageVersion("spacexr")), "\n"); print(.libPaths())'
