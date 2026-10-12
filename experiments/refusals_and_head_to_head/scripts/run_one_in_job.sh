#!/bin/bash
# Run one task of a task file inside a running H2H job allocation (same environment as the job script).
# Usage: run_one_in_job.sh JOBID PORT TASKS_NAME INDEX
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}; B=$HOME/expBX; G=$B/mirror; X=$G$S/expBX
exec srun --jobid=$1 --overlap -n 1 -c 4 env EXPBX_BACKEND=openai EXPBX_MODEL=Qwen3.8-27B-FP8 EXPBX_BASE_URL=http://127.0.0.1:$2/v1 \
  EXPBX_HOST_PREFIX=$G EXPBX_CONTAINER=docker EXPBX_DOCKER_IMAGE=buildpack-deps:jammy-scm EXPBX_CS_ENV=$B/envs/chatspatial_bx159 \
  EXPBX_UV_PY=$HOME/.local/share/uv/python/cpython-3.11.14-linux-aarch64-gnu EXPBX_LOCAL=/tmp SLURM_CPUS_PER_TASK=4 \
  EXPBX_DOCKER_EXTRA="--oom-score-adj 800" /usr/bin/python3 $X/scripts/expBX_run_task.py $X/tasks/$3.jsonl $4
