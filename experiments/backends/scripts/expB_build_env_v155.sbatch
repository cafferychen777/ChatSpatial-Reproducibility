#!/bin/bash
#SBATCH -J expB_v155rebuild
#SBATCH -p medium
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o expB_logs/%x_%j.out
# chatspatial_v155 and chatspatial_expB were removed from conda_envs by 2026-10-09 19:35 (not by this harness).
# Rebuild chatspatial_v155 the same way: python 3.12 + conda-forge R 4.4.3, chatspatial[full]==1.5.5, rpy2 3.5.17;
# R packages come from the existing no_user_rlib / rlib_v155 libraries (built for R 4.4.3).
set -uo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}
E=$S/conda_envs/chatspatial_v155
source ~/miniconda3/etc/profile.d/conda.sh
export TMPDIR=$S/expB/tmp PYTHONNOUSERSITE=1
conda create -y -p $E -c conda-forge -c bioconda --override-channels \
  python=3.12 r-base=4.4.3 r-remotes compilers make cmake pkg-config hdf5 \
  bioconductor-biobase bioconductor-singlecellexperiment bioconductor-summarizedexperiment bioconductor-biocgenerics \
  bioconductor-complexheatmap bioconductor-biocneighbors r-nmf r-circlize r-ggalluvial r-sna r-svglite r-reticulate \
  r-doparallel r-quadprog r-irlba r-fnn r-ggpubr r-ggnetwork r-patchwork r-cowplot r-mcmcpack r-fields r-concaveman \
  r-rcppml r-pbmcapply r-rcpparmadillo r-compquadform r-pracma r-matlab r-rspectra r-future r-future.apply r-bh \
  r-plotly r-ggrepel r-stringr r-expm r-shape r-seurat r-gtools r-wrapr r-ggrastr r-scales r-dplyr r-tidyr >/dev/null \
  || { echo CREATE_FAILED; exit 2; }
conda activate $E
python -m pip install -q --upgrade pip
python -m pip install "chatspatial[full]==1.5.5" "numpy<2.4" 2>&1 | tail -2
python -m pip install "rpy2==3.5.17" 2>&1 | tail -1
export R_LIBS_USER=$S/expB/rlib_v155:$S/expB/no_user_rlib R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
R --version | head -1
python -c "import chatspatial, rpy2, numpy; print('chatspatial', chatspatial.__version__, 'rpy2', rpy2.__version__, 'numpy', numpy.__version__)"
python -c "import rpy2.robjects as ro; ro.r('suppressMessages({library(spacexr); library(CellChat); library(SPARK); library(CARD); library(brglm2); library(lme4)})'); print('R pkgs OK')" 2>&1 | tail -1
for m in STAGATE_pyG GraphST SpaGCN cellrank palantir infercnvpy liana SpatialDE scvelo fastccc cellphonedb flashdeconv banksy; do python -W ignore -c "import $m" >/dev/null 2>&1 && echo "PY $m ok" || echo "PY $m MISSING"; done
python -m pip freeze > $S/expB/logs/pip_freeze_chatspatial_v155_rebuilt.txt
echo BUILD_DONE
