#!/bin/bash
#SBATCH --job-name=expA_env155
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_env155_%j.log
# Experiment-A environment from the released packages: chatspatial[full]==1.5.5, spagcn-modern 1.4.0.
set -euo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}
ROOT=$S/chatspatial_expA
ENV=$S/conda_envs/chatspatial_expA_155
module load CMake/3.27.6-GCCcore-13.2.0
eval "$(conda shell.bash hook)"
[ -x "$ENV/bin/python" ] || conda create -y -p "$ENV" python=3.12 pip
conda activate "$ENV"
export PIP_NO_CACHE_DIR=1 CMAKE_BUILD_PARALLEL_LEVEL=8
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "chatspatial[full]==1.5.5" "spagcn-modern==1.4.0"
# louvain backend (sc.tl.louvain, GraphST 'louvain'); its metadata pins igraph<0.12, so no deps
python -m pip install --no-deps louvain==0.8.2
python - <<'PY'
import importlib, importlib.metadata as md
for p in ["chatspatial", "spagcn-modern", "stagate-modern", "graphst-modern", "pybanksy", "torch",
          "torch-geometric", "scanpy", "squidpy", "louvain", "leidenalg", "igraph", "anndata", "numpy"]:
    try: print(p, md.version(p))
    except Exception as e: print(p, "MISSING")
for m in ["chatspatial.tools.spatial_domains", "SpaGCN", "STAGATE_pyG", "GraphST", "banksy", "louvain"]:
    try: importlib.import_module(m); print("import ok", m)
    except Exception as e: print("import FAILED", m, e)
from chatspatial.models.data import SpatialDomainParameters as S
print("spagcn_compat default:", S().spagcn_compat)
PY
python -m pip freeze > "$ROOT/logs/pip_freeze_155.txt"
echo DONE
