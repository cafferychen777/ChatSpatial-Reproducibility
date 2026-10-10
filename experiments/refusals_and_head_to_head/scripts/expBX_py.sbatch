#!/bin/bash
#SBATCH -J bx_py
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:55:00
#SBATCH -o logs/%x_%j.out
# Run an expBX python script in the ChatSpatial v155 env (judge, analysis). Usage: sbatch expBX_py.sbatch SCRIPT [ARGS...]
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}
V=$S/conda_envs/chatspatial_v155
export PATH=$V/bin:/usr/bin:/bin PYTHONNOUSERSITE=1 MPLBACKEND=Agg EXPBX_ROOT=$S/expBX
export PYTHONPATH=$S/expBX/pylib:$S/expB/compat:$S/expBX/scripts
cd $S/expBX
python -W ignore "$S/expBX/scripts/$@"
