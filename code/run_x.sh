#!/bin/bash
# bridge-resistance stress, horizon scaling at the 2000-evaluation budget, corner replication. Output: $OUT/results, $OUT/logs
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
( for r in 0.2 0.1 0.05; do RAB=$r $PYTHON sens_run.py "$OUT/results/rab_$r.jsonl" >> "$OUT/logs/rab_$r.log" 2>&1 || echo "FAILED rab $r" >> "$OUT/logs/failures.txt"; done ) &
( for k in 0.8 1.2; do KT=$k $PYTHON sens_run.py "$OUT/results/T_$k.jsonl" >> "$OUT/logs/T_$k.log" 2>&1 || echo "FAILED T $k" >> "$OUT/logs/failures.txt"; done
  for c in 0.8,0.8,0.8 1.2,0.8,0.8; do SENS=$c $PYTHON sens_corner.py "$OUT/results/corner_${c//,/_}.jsonl" >> "$OUT/logs/corner_$c.log" 2>&1 || echo "FAILED corner $c" >> "$OUT/logs/failures.txt"; done ) &
wait
finish x
