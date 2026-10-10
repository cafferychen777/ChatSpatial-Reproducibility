#!/bin/bash
#SBATCH --job-name=expB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=expB_logs/%x_%j.log
# Experiment B (LLM backend dependence): one vLLM OpenAI-compatible server on 127.0.0.1 inside
# this job, then DRIVER_CMD against it in the same job. Nothing listens on a public interface.
#
# Usage (from experiments/backends/scripts; the open-weight arms used expB_ow_driver.sh as DRIVER_CMD):
#   MODEL_KEY=gemma4-12b DRIVER_CMD='bash expB_ow_driver.sh' sbatch --job-name=expB-gemma4 expB_vllm_job.sbatch
#   MODEL_KEY=qwen38-27b DRIVER_CMD='bash expB_ow_driver.sh' sbatch --job-name=expB-qwen38 expB_vllm_job.sbatch
# DRIVER_CMD is eval'ed after the server is healthy, with cwd = the submit directory and these exported:
#   BASE_URL (http://127.0.0.1:PORT/v1), SERVED_NAME, PORT, RUN_DIR (per-job output dir).
# Any preset value can be overridden by exporting MODEL / REVISION / SERVED_NAME / SERVE_ARGS /
# MAX_MODEL_LEN before sbatch (sbatch exports the caller's environment by default).
# Outputs: $RUN_DIR/vllm_server.log and $RUN_DIR/driver.log.
set -uo pipefail
W=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}/expB
MODEL_KEY=${MODEL_KEY:-}

# Presets verified by the 2026-10-08 smoke tests (vLLM 0.31.0, 1x H100 80GB).
# Thinking is off for both (single tool call per request); the reasoning parsers only strip
# the empty thought block the templates still emit.
case "$MODEL_KEY" in
  gemma4-12b)
    : "${MODEL:=google/gemma-4-12B-it}"
    : "${REVISION:=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"
    : "${SERVED_NAME:=gemma-4-12B-it}"
    : "${MAX_MODEL_LEN:=32768}"
    [ -n "${SERVE_ARGS:-}" ] || SERVE_ARGS='--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --limit-mm-per-prompt {"image":0,"audio":0}'
    ;;
  qwen38-27b)
    : "${MODEL:=Qwen/Qwen3.8-27B-FP8}"
    : "${REVISION:=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}"
    : "${SERVED_NAME:=Qwen3.8-27B-FP8}"
    : "${MAX_MODEL_LEN:=32768}"
    # --max-num-seqs 64: CUDA-graph capture needs one Mamba cache block per sequence.
    [ -n "${SERVE_ARGS:-}" ] || SERVE_ARGS='--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --default-chat-template-kwargs {"enable_thinking":false} --max-num-seqs 64 --limit-mm-per-prompt {"image":0,"video":0}'
    ;;
  "") ;;  # fully custom: MODEL, REVISION, SERVED_NAME, SERVE_ARGS, MAX_MODEL_LEN must be exported
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
: "${MODEL:?}" "${REVISION:?}" "${SERVED_NAME:?}" "${SERVE_ARGS:?}" "${MAX_MODEL_LEN:=32768}"
: "${DRIVER_CMD:?set DRIVER_CMD}"
: "${GPU_MEM_UTIL:=0.90}"

RUN_DIR=$W/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR" $W/cache/{triton,vllm,inductor} $W/tmp

module purge; module load GCCcore/13.3.0 Python/3.12.3
source $W/venv/bin/activate
export HF_HOME=${HF_HOME:-$W/hf} HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$W/cache TRITON_CACHE_DIR=$W/cache/triton VLLM_CACHE_ROOT=$W/cache/vllm \
       TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor TMPDIR=$W/tmp PYTHONUNBUFFERED=1
# DeepGEMM's JIT FP8 kernels fail to compile on ACES; use vLLM's CUTLASS FP8 path.
export VLLM_USE_DEEP_GEMM=0
# FlashInfer top-k/top-p sampler JIT-compiles with nvcc, which is not on the node; use the PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
export MODEL PORT SERVED_NAME RUN_DIR BASE_URL=http://127.0.0.1:$PORT/v1
{
  echo "job=$SLURM_JOB_ID node=$(hostname) date=$(date -Is)"
  echo "model=$MODEL revision=$REVISION served=$SERVED_NAME max_model_len=$MAX_MODEL_LEN"
  echo "serve_args=$SERVE_ARGS"
  echo "vllm=$(python -c 'import vllm; print(vllm.__version__)')"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "driver_cmd=$DRIVER_CMD"
} | tee "$RUN_DIR/job_info.txt"

# SERVE_ARGS is word-split on purpose; JSON values in it must contain no spaces.
vllm serve "$MODEL" --revision "$REVISION" --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" $SERVE_ARGS \
  > "$RUN_DIR/vllm_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT

for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "SERVER_DEAD"; tail -40 "$RUN_DIR/vllm_server.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "SERVER_TIMEOUT"; tail -40 "$RUN_DIR/vllm_server.log"; exit 1; }
echo "SERVER_READY after ~$((i * 10))s at $BASE_URL"

cd ${EXPB_SCRIPTS:-$SLURM_SUBMIT_DIR}
eval "$DRIVER_CMD" > "$RUN_DIR/driver.log" 2>&1
rc=$?
echo "DRIVER_EXIT=$rc  (log: $RUN_DIR/driver.log)"
tail -n 60 "$RUN_DIR/driver.log"
exit $rc
