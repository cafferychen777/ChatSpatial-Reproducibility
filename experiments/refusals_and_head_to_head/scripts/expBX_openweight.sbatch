#!/bin/bash
#SBATCH -J bx_ow
#SBATCH -p gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 20
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.out
# Experiment B-X open-weight arm on a single-GPU host (NVIDIA GB10, aarch64): vLLM 0.31.0 on 127.0.0.1 and the probe runs of all
# three systems in Docker containers with read-only binds (rules section 6). Usage: sbatch expBX_openweight.sbatch <model_key>
set -uo pipefail
KEY=${1:?model key}
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}  # path of the CPU-cluster work directory, mirrored below
B=$HOME/expBX
G=$B/mirror
X=$G$S/expBX
case $KEY in
 gemma4-12b) MODEL=google/gemma-4-12B-it; REV=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; NAME=gemma-4-12B-it; TAG=ow_gemma4
   ARGS='--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --limit-mm-per-prompt {"image":4,"audio":0}' ;;
 qwen38-27b) MODEL=Qwen/Qwen3.8-27B-FP8; REV=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a; NAME=Qwen3.8-27B-FP8; TAG=ow_qwen38
   ARGS='--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --default-chat-template-kwargs {"enable_thinking":false} --max-num-seqs 64 --limit-mm-per-prompt {"image":4,"video":0}' ;;
 *) echo unknown; exit 2 ;;
esac
CONC=${CONC:-4}
LOG=$B/logs/ow_${TAG}_$SLURM_JOB_ID; mkdir -p $LOG
source $B/envs/vllm/bin/activate
export HF_HOME=$B/hf HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=$B/cache VLLM_CACHE_ROOT=$B/cache/vllm TRITON_CACHE_DIR=$B/cache/triton
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
BASE_URL=http://127.0.0.1:$PORT/v1
{ echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date -Is) model=$MODEL rev=$REV served=$NAME len=131072"; echo "args=$ARGS";
  echo "vllm=$(python -c 'import vllm;print(vllm.__version__)')"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; id -nG; } | tee $LOG/job_info.txt
vllm serve $MODEL --revision $REV --served-model-name $NAME --host 127.0.0.1 --port $PORT --max-model-len 131072 \
  --gpu-memory-utilization ${GPU_UTIL:-0.45} $ARGS > $LOG/vllm_server.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SPID || { echo SERVER_DEAD; tail -40 $LOG/vllm_server.log; exit 1; }; sleep 10; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo SERVER_TIMEOUT; exit 1; }
echo "SERVER_READY $BASE_URL"
deactivate
export EXPBX_BACKEND=openai EXPBX_MODEL=$NAME EXPBX_BASE_URL=$BASE_URL EXPBX_HOST_PREFIX=$G EXPBX_CONTAINER=docker \
       EXPBX_DOCKER_IMAGE=buildpack-deps:jammy-scm EXPBX_CS_ENV=$B/envs/chatspatial_bx156 \
       EXPBX_UV_PY=$HOME/.local/share/uv/python/cpython-3.11.14-linux-aarch64-gnu EXPBX_LOCAL=/tmp SLURM_CPUS_PER_TASK=4
RT=$X/scripts/expBX_run_task.py
PT=$X/tasks/${TAG}_pilot.jsonl; FT=$X/tasks/${TAG}_probe.jsonl
seq 0 $(( $(wc -l < $PT) - 1 )) | xargs -P $CONC -I{} /usr/bin/python3 $RT $PT {} > $LOG/pilot.log 2>&1
/usr/bin/python3 - "$PT" "$G" <<'PY' || { echo PILOT_GATE_FAILED; exit 3; }
import json, sys, os
G = sys.argv[2]; sys.path.insert(0, G + os.environ["CHATSPATIAL_WORK_DIR"] + "/expBX/scripts")
import expBX_common as C
ok = n = 0
for l in open(sys.argv[1]):
    t = json.loads(l); n += 1
    rd = G + str(C.run_dir(t))
    log = rd + ("/transcript.jsonl" if t["system"] == "chatspatial" else "/api_log.jsonl")
    got = os.path.exists(log) and any('"api_response"' in x for x in open(log))
    res = os.path.exists(rd + "/result.json")
    print(t["task_id"], "api_response" if got else "NO_API_RESPONSE", "result" if res else "NO_RESULT")
    ok += bool(got and res)
print(f"PILOT {ok}/{n}")
sys.exit(0 if ok >= 1 else 1)
PY
echo PILOT_PASSED $(date -Is)
seq 0 $(( $(wc -l < $FT) - 1 )) | xargs -P $CONC -I{} /usr/bin/python3 $RT $FT {} > $LOG/full.log 2>&1
echo FULL_DONE $(date -Is)
