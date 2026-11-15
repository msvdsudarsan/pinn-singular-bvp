#!/bin/bash
# independent resistance-group scalings. Output: $OUT/results, $OUT/logs
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run() { for g in "$@"; do GRP=$g $PYTHON sens_run.py "$OUT/results/grp_${g//,/_}.jsonl" >> "$OUT/logs/grp_${g//,/_}.log" 2>&1 || echo "FAILED grp $g" >> "$OUT/logs/failures.txt"; done; }
run 0.8,1,1 1,0.8,1 1,1,0.8 & run 1.2,1,1 1,1.2,1 1,1,1.2 & wait
finish grp
