#!/bin/bash
# Usage (on a compute node): final_eval.sh TAG   -> results/eval_TAG.txt
set -uo pipefail
cd "${CHATSPATIAL_WORK_DIR:?set CHATSPATIAL_WORK_DIR}/chatspatial_expA/scripts"
for s in dlpfc155 noise155 platforms155; do python incomplete_tasks.py $s; done
python evaluate.py dlpfc155 noise155 platforms155 > ../results/eval_$1.txt 2>&1
python pooled.py dlpfc155 platforms155 >> ../results/eval_$1.txt 2>&1
cat ../results/eval_$1.txt
