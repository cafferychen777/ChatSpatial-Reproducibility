#!/bin/bash
#SBATCH -J mdeconv_music
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
# CARD calls library(MuSiC), which the chatspatial_v155 env lacks. Install it into a separate library;
# existing envs and libraries are not modified.
set -uo pipefail
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR to the cluster work directory}; W=$S/mdeconv
eval "$(conda shell.bash hook)"
conda activate $S/conda_envs/chatspatial_v155
mkdir -p $W/rlib_music
export R_LIBS_USER=$W/rlib_music:$S/expB/rlib_v155:$S/expB/no_user_rlib R_LIBS= R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null \
       R_MAKEVARS_USER=$S/expB/Makevars_cxx17 TMPDIR=$W/tmp MAKEFLAGS=-j4
R --version | head -1
Rscript -e '
lib <- Sys.getenv("R_LIBS_USER"); lib <- strsplit(lib, ":")[[1]][1]
options(repos=c(CRAN="https://cloud.r-project.org"), Ncpus=4)
need <- c("nnls","Biobase","SingleCellExperiment","TOAST","remotes","BiocManager")
miss <- need[!vapply(need, requireNamespace, logical(1), quietly=TRUE)]
cat("missing before:", miss, "\n")
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager", lib=lib)
bioc <- intersect(miss, c("Biobase","SingleCellExperiment","TOAST"))
if (length(bioc)) BiocManager::install(bioc, lib=lib, update=FALSE, ask=FALSE)
cran <- intersect(miss, c("nnls","remotes"))
if (length(cran)) install.packages(cran, lib=lib)
remotes::install_github("xuranw/MuSiC", lib=lib, upgrade="never", dependencies=TRUE, build_vignettes=FALSE)
suppressMessages({library(MuSiC); library(CARD)})
cat("MuSiC", as.character(packageVersion("MuSiC")), "CARD", as.character(packageVersion("CARD")), "OK\n")'
echo MUSIC_DONE
