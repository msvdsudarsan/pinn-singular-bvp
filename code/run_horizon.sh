#!/bin/bash
# horizon x budget study: KT in {0.8,1.0,1.2,1.5} gives T = 9.6, 12, 14.4, 18 ms; budgets 2000/4000/8000 are set inside sens_horizon.py
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for k in 0.8 1.0 1.2 1.5; do KT=$k $PYTHON sens_horizon.py "$OUT/results/horizon_T_$k.jsonl" >> "$OUT/logs/horizon_T_$k.log" 2>&1 || echo "FAILED horizon $k" >> "$OUT/logs/failures.txt"; done
finish horizon
