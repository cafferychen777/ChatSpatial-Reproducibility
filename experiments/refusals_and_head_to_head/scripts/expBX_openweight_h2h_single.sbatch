#!/bin/bash
#SBATCH -J bx_ow_h2h_single
#SBATCH -p gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 20
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.out
# Qwen3.8-27B-FP8 arm of the DLPFC 12-section head-to-head (rules addendum h2h_qwen_rules_2026-10-10.txt) on
# the single-GPU host (NVIDIA GB10, aarch64). vLLM 0.31.0 on 127.0.0.1 with exactly the serve settings of the Qwen probe arm (job 119);
# runs of chatspatial (1.5.9 env), stagent and spatialagent in Docker with read-only binds (as the probe arm).
# Copy of expBX_openweight_h2h.sbatch (the file used for jobs 121-125) for the chatspatial_single Qwen arm
# (rules addendum of 2026-10-11): identical server and run settings; the only addition is the GATE check at the end.
# GATE=1: exit 1 unless every task of the file finished with success and labels.csv (smoke gate for an afterok chain).
# Usage: [GATE=1] sbatch expBX_openweight_h2h_single.sbatch TASKS_NAME     (tasks/<TASKS_NAME>.jsonl; resumable: finished runs skipped)
# A run is started only if >= 47 min remain in the job, so no run is cut by the job time limit; the next chained job
# (afterany dependency) continues the queue.
set -uo pipefail
TASKS=${1:?tasks name}
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}  # path of the CPU-cluster work directory, mirrored below
B=$HOME/expBX
G=$B/mirror
X=$G$S/expBX
MODEL=Qwen/Qwen3.8-27B-FP8; REV=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a; NAME=Qwen3.8-27B-FP8
ARGS='--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --default-chat-template-kwargs {"enable_thinking":false} --max-num-seqs 64 --limit-mm-per-prompt {"image":4,"video":0}'
CONC=${CONC:-4}
DEADLINE=$(date -d "$(scontrol show job $SLURM_JOB_ID | grep -oP 'EndTime=\K\S+')" +%s 2>/dev/null) || DEADLINE=0
[ "${DEADLINE:-0}" -gt 0 ] || DEADLINE=$(( $(date +%s) + 24*3600 ))
DEADLINE=$(( DEADLINE - 120 ))
LOG=$B/logs/ow_h2h_${TASKS}_$SLURM_JOB_ID; mkdir -p $LOG
source $B/envs/vllm/bin/activate
export HF_HOME=$B/hf HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=$B/cache VLLM_CACHE_ROOT=$B/cache/vllm TRITON_CACHE_DIR=$B/cache/triton
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
BASE_URL=http://127.0.0.1:$PORT/v1
{ echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date -Is) tasks=$TASKS model=$MODEL rev=$REV served=$NAME len=131072 conc=$CONC"
  echo "args=$ARGS"; echo "vllm=$(python -c 'import vllm;print(vllm.__version__)')"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } | tee $LOG/job_info.txt
vllm serve $MODEL --revision $REV --served-model-name $NAME --host 127.0.0.1 --port $PORT --max-model-len 131072 \
  --gpu-memory-utilization ${GPU_UTIL:-0.45} $ARGS > $LOG/vllm_server.log 2>&1 &
SPID=$!
cleanup() {
  docker ps -q --filter name=expbx_ | xargs -r docker kill >/dev/null 2>&1
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; [ -n "${PPID_LOOP:-}" ] && kill $PPID_LOOP 2>/dev/null
}
trap cleanup EXIT
for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SPID || { echo SERVER_DEAD; tail -40 $LOG/vllm_server.log; exit 1; }; sleep 10; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo SERVER_TIMEOUT; exit 1; }
echo "SERVER_READY $BASE_URL $(date -Is)"
deactivate
export EXPBX_BACKEND=openai EXPBX_MODEL=$NAME EXPBX_BASE_URL=$BASE_URL EXPBX_HOST_PREFIX=$G EXPBX_CONTAINER=docker \
       EXPBX_DOCKER_IMAGE=buildpack-deps:jammy-scm EXPBX_CS_ENV=$B/envs/chatspatial_bx159 \
       EXPBX_UV_PY=$HOME/.local/share/uv/python/cpython-3.11.14-linux-aarch64-gnu EXPBX_LOCAL=/tmp SLURM_CPUS_PER_TASK=4 \
       EXPBX_DOCKER_EXTRA="--oom-score-adj 800" DEADLINE RT=$X/scripts/expBX_run_task.py FT=$X/tasks/$TASKS.jsonl
# progress file, refreshed every 10 min
( while true; do /usr/bin/python3 $X/scripts/expBX_ow_h2h_progress.py $TASKS > $LOG/progress.txt 2>&1
    cp $LOG/progress.txt $B/logs/progress_$TASKS.txt; sleep 600; done ) &
PPID_LOOP=$!
N=$(wc -l < $FT)
seq 0 $((N - 1)) | xargs -P $CONC -I{} bash -c '
  if [ $(( DEADLINE - $(date +%s) )) -lt 2820 ]; then echo "SKIP_DEADLINE {}"; exit 0; fi
  /usr/bin/python3 $RT $FT {}' >> $LOG/runs.log 2>&1
/usr/bin/python3 $X/scripts/expBX_ow_h2h_progress.py $TASKS > $LOG/progress.txt 2>&1; cp $LOG/progress.txt $B/logs/progress_$TASKS.txt
echo "QUEUE_PASS_DONE $(date -Is) skipped_for_deadline=$(grep -c SKIP_DEADLINE $LOG/runs.log)"
if [ "${GATE:-0}" = 1 ]; then
  /usr/bin/python3 - "$X" "$FT" <<'PYEOF' || { echo GATE_FAILED; exit 1; }
import json, sys
from pathlib import Path
X, FT = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(X / "scripts"))
import os
import expBX_common as C
G = os.path.expanduser("~/expBX/mirror")
ok = True
for t in (json.loads(l) for l in open(FT) if l.strip()):
    rd = Path(G + str(C.run_dir(t)))
    try:
        st, r = json.loads((rd / "status.json").read_text()), json.loads((rd / "result.json").read_text())
    except Exception as e:
        print("GATE", t["task_id"], "missing status/result", e); ok = False; continue
    good = st.get("finished") and not st.get("timed_out") and r.get("success") and (rd / "labels.csv").exists()
    print("GATE", t["task_id"], "ok" if good else "FAIL", r.get("method"), r.get("n_labels"), r.get("error"),
          round((st.get("wall_time") or 0) / 60, 1), "min")
    ok = ok and bool(good)
sys.exit(0 if ok else 1)
PYEOF
  echo GATE_PASSED
fi
