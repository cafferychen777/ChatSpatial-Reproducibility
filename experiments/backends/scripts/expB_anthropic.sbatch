#!/bin/bash
#SBATCH -J expB_anth
#SBATCH -p medium
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH -o expB_logs/%x_%A_%a.out
# Experiment B, Anthropic backends (LLM calls only). Resumable: re-submitting skips finished trials.
# Usage: sbatch expB_anthropic.sbatch [EXPERIMENTS] [EXTRA_ARGS]
#   EXPERIMENTS default "main desc bias"; pilot: sbatch expB_anthropic.sbatch "main desc bias" "--prompts spatial_domains,bias_domains --reps 2 --pilot"
# The API key is read at runtime by expB_invocation.py from $ANTHROPIC_API_KEY or ~/.config/chatspatial/anthropic_key.
# Submit from experiments/backends/scripts.
set -euo pipefail
MODELS=(claude-haiku-4-5-20251001 claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
EXPS=${1:-"main desc bias"}
EXTRA=${2:-""}
module load Python/3.11.5-GCCcore-13.2.0 >/dev/null 2>&1
S=${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}
cd ${EXPB_SCRIPTS:-$SLURM_SUBMIT_DIR}
OUTDIR=$S/expB/invocation; mkdir -p $OUTDIR
SUFFIX=""
if [[ "$EXTRA" == *"--pilot"* ]]; then SUFFIX="__pilot"; EXTRA=${EXTRA/--pilot/}; fi
echo "$(date -Is) host=$(hostname) model=$M exps=$EXPS extra=$EXTRA"
for E in $EXPS; do
  python3 -u expB_invocation.py --backend anthropic --model "$M" --experiment "$E" --workers 4 $EXTRA \
      --out "$OUTDIR/${M}__${E}${SUFFIX}.jsonl"
done
echo "$(date -Is) finished"
