#!/bin/bash
# In-job driver for experiment B open-weight backends (run by expB_vllm_job.sbatch after the server is healthy).
# Needs BASE_URL, SERVED_NAME, MODEL (HF repo id) from the job. Pilot first (1 prompt x all cells, 2 reps);
# the full run starts only if every pilot call returned without an API error and >= 50% parsed.
set -uo pipefail
cd "$(dirname "$0")"
OUT=${OW_ROOT:-${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}/expB}/invocation; mkdir -p $OUT
LABEL="ow:${MODEL}"; SHORT=$(echo "$MODEL" | tr '/:' '__')
for E in main desc bias; do
  python expB_invocation.py --backend openai_compat --base-url "$BASE_URL" --model "$SERVED_NAME" --label "$LABEL" \
    --experiment $E --prompts spatial_domains,bias_domains --reps 2 --workers 8 --out $OUT/${SHORT}__${E}__pilot.jsonl
done
python - "$OUT" "$SHORT" <<'PY' || { echo "PILOT_GATE_FAILED"; exit 3; }
import json, sys, glob
recs = [json.loads(l) for f in glob.glob(f"{sys.argv[1]}/{sys.argv[2]}__*__pilot.jsonl") for l in open(f)]
err = sum(r["error"] is not None for r in recs); ok = sum(r.get("parse_status") == "ok" for r in recs)
print(f"PILOT n={len(recs)} api_errors={err} parse_ok={ok}")
for r in recs[:6]:
    print(" ", r["mechanism"], r["info"], r["arm"], r.get("parse_status"), r.get("canonical_tool"), json.dumps(r.get("params"))[:120])
sys.exit(0 if recs and err == 0 and ok >= 0.5 * len(recs) else 1)
PY
echo PILOT_GATE_PASSED
for E in main desc bias; do
  python expB_invocation.py --backend openai_compat --base-url "$BASE_URL" --model "$SERVED_NAME" --label "$LABEL" \
    --experiment $E --workers 8 --out $OUT/${SHORT}__${E}.jsonl
done
echo FULL_DONE
