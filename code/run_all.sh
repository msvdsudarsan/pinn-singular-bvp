#!/bin/bash
# Master script: every computational study reported in the manuscript and supplement.
#   ./run_all.sh quick   (default) one representative seed (two for extension, enclosure studies) of each study family listed below, compared record by record with the shipped results. Labelled PARTIAL: a subset, not all configurations or seeds.
#   ./run_all.sh full    all studies, all seeds (about 7 CPU-hours of training; roughly 4 h wall time on 2 cores), then check_paper_values.py on the new results. Labelled FULL.
# Needs: SUPP=/path/to/supplement.tex (the proof-verification script checks one constant against the supplement source).
# New records go to $OUT (default <repo>/rerun_output); the shipped results/ folder is never written.
# Exit status 0 only if every step ended with status 0 and printed PASS. The final label (FULL or PARTIAL) is written to $OUT/logs/RUN_LABEL.txt.
MODE="${1:-quick}"
case "$MODE" in quick|full) ;; *) echo "usage: $0 [quick|full]" >&2; exit 2;; esac
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
RES="$ROOT/results"; LOG="$OUT/logs"; SUMMARY="$LOG/summary.tsv"; [ -e "$LOG/done" ] || : > "$SUMMARY"   # a resumed run keeps its earlier summary lines
if [ -e "$OUT/results/.started" ]; then echo "WARNING: $OUT/results already holds records from an earlier run; existing records are resumed or appended, not overwritten. Use a fresh OUT= for a clean run." >&2; fi
touch "$OUT/results/.started"
SUPP="${SUPP:-}"
T0=$(date +%s)
# versions, seeds, commands
{ echo "date: $(date -u +%FT%TZ)"; echo "mode: $MODE"; echo "host: $(uname -srm)"; echo "cpus: $(nproc)"
  $PYTHON - <<'PY'
import sys, platform, importlib
print("python", sys.version.split()[0])
for m in ("jax", "jaxlib", "numpy", "scipy", "mpmath", "sympy", "pandas"):
    try: print(m, importlib.import_module(m).__version__)
    except Exception as e: print(m, "MISSING", e)
import jax; print("jax x64 default:", jax.config.jax_enable_x64, "(scripts enable it explicitly); devices:", jax.devices())
PY
  echo "seeds: seed values are the loop indices 0..9 inside the scripts (jax PRNGKey(seed)); networks 1->32->32->32->nout tanh unless an ablation changes it"; } > "$LOG/environment.txt"
mkdir -p "$LOG/done"
step() {  # step NAME COMMAND...   : runs a command, logs it, records exit code and runtime. A step that already finished with exit 0 in this OUT is skipped (resume after an interruption); failed steps are re-run.
  local name="$1"; shift; local t=$(date +%s)
  # optional split over several processes (same OUT): ONLY_RE runs only matching steps, SKIP_RE skips matching steps; a final plain run then skips all completed steps and writes the verdict
  if [ -n "$ONLY_RE" ] && ! [[ "$name" =~ $ONLY_RE ]]; then return 0; fi
  if [ -n "$SKIP_RE" ] && [[ "$name" =~ $SKIP_RE ]]; then return 0; fi
  if [ -e "$LOG/done/$name" ]; then echo "[$(date +%T)] SKIP  $name (completed earlier in this output directory)" | tee -a "$LOG/commands.txt"; return 0; fi
  echo "[$(date +%T)] START $name: $*" | tee -a "$LOG/commands.txt"
  "$@" > "$LOG/$name.log" 2>&1; local rc=$?
  printf "%s\t%d\t%d\n" "$name" "$rc" "$(( $(date +%s) - t ))" >> "$SUMMARY"
  if [ $rc -ne 0 ]; then echo "FAILED $name (exit $rc)" >> "$LOG/failures.txt"; else touch "$LOG/done/$name"; [ -f "$LOG/failures.txt" ] && sed -i "/^FAILED $name (/d" "$LOG/failures.txt"; fi
  echo "[$(date +%T)] END   $name exit=$rc" | tee -a "$LOG/commands.txt"; return 0
}
cmp_() {  # cmp_ NAME RERUN SHIPPED [rtol]
  step "compare_$1" $PYTHON compare_rerun.py "$2" "$3" ${4:-1e-6}
}
jl() { grep '^{' "$1" > "$2"; }   # stdout of the original-example scripts -> jsonl

# ---- 0 mathematical checks and validation
if [ -n "$SUPP" ]; then step proof_verification $PYTHON proof_verification.py "$SUPP"
else echo "FAILED proof_verification (set SUPP=/path/to/supplement.tex)" >> "$LOG/failures.txt"; printf "proof_verification\t2\t0\n" >> "$SUMMARY"; fi
SENS=1,1,1 step mna_nominal $PYTHON mna_validate.py
step cert_kd_check $PYTHON cert_kd_check.py
step extA_landscape $PYTHON extA_landscape.py
SEEDQ="0 1"; [ "$MODE" = full ] && SEEDQ="0 1 2 3 4"   # cert.py: the shipped records use seeds 0-4 (nominal) and 0-2 (failing corner)
SEEDX="0 1"
NS=1; [ "$MODE" = full ] && NS=10

# ---- 1 original four examples: configuration sweep (6 configs x 4 examples), switch-point sweep, activation/width ablation
S1=1; [ "$MODE" = full ] && S1=10
step orig_examples $PYTHON original_examples_pinn.py 0 $S1; jl "$LOG/orig_examples.log" "$OUT/results/orig_examples.jsonl"
cmp_ orig_examples "$OUT/results/orig_examples.jsonl" "$RES/original_examples_pinn_results.csv"
# switch-point sweep and activation/width ablation (the script resumes from log_sweep.txt; it is cleared so that nothing is skipped silently)
: > log_sweep.txt
SW=1; AB=1; [ "$MODE" = full ] && { SW=10; AB=4; }
NSEED=$SW step orig_sweep $PYTHON original_examples_sweep_ablation.py sweep; jl "$LOG/orig_sweep.log" "$OUT/results/orig_sweep.jsonl"; cmp_ orig_sweep "$OUT/results/orig_sweep.jsonl" "$RES/switch_sweep_results.csv"
if [ "$MODE" = full ]; then ABS=2; else ABS=1; fi
step orig_ablate $PYTHON original_examples_sweep_ablation.py ablate 0.3 $AB $ABS; jl "$LOG/orig_ablate.log" "$OUT/results/orig_ablate.jsonl"; cmp_ orig_ablate "$OUT/results/orig_ablate.jsonl" "$RES/activation_width_results.csv"

# ---- 2 RLC network: nominal, 14 sensitivity cases, group scalings, bridge-resistance stress, horizon scaling, corners
if [ "$MODE" = full ]; then
  while read -r c; do SENS=$c NSEED=$NS step "case_$c" $PYTHON sens_run.py "$OUT/results/case_${c//,/_}.jsonl"; cmp_ "case_$c" "$OUT/results/case_${c//,/_}.jsonl" "$RES/sensitivity_runs/case_${c//,/_}.jsonl"; done < cases.txt
  for g in 0.8,1,1 1.2,1,1 1,0.8,1 1,1.2,1 1,1,0.8 1,1,1.2; do GRP=$g step "grp_$g" $PYTHON sens_run.py "$OUT/results/grp_${g//,/_}.jsonl"; cmp_ "grp_$g" "$OUT/results/grp_${g//,/_}.jsonl" "$RES/group_runs/grp_${g//,/_}.jsonl"; done
  for r in 0.2 0.1 0.05; do RAB=$r step "rab_$r" $PYTHON sens_run.py "$OUT/results/rab_$r.jsonl"; cmp_ "rab_$r" "$OUT/results/rab_$r.jsonl" "$RES/stress_runs/rab_$r.jsonl"; done
  for k in 0.8 1.2; do KT=$k step "T_$k" $PYTHON sens_run.py "$OUT/results/T_$k.jsonl"; cmp_ "T_$k" "$OUT/results/T_$k.jsonl" "$RES/stress_runs/T_$k.jsonl"; done
  for c in 0.8,0.8,0.8 1.2,0.8,0.8; do SENS=$c step "corner_$c" $PYTHON sens_corner.py "$OUT/results/corner_${c//,/_}.jsonl"; cmp_ "corner_$c" "$OUT/results/corner_${c//,/_}.jsonl" "$RES/stress_runs/corner_${c//,/_}.jsonl"; done
  for k in 0.8 1.0 1.2 1.5; do KT=$k step "horizon_$k" $PYTHON sens_horizon.py "$OUT/results/horizon_T_$k.jsonl"; cmp_ "horizon_$k" "$OUT/results/horizon_T_$k.jsonl" "$RES/results_h/T_$k.jsonl"; done
else
  SENS=1,1,1 NSEED=1 step case_nominal $PYTHON sens_run.py "$OUT/results/case_1_1_1.jsonl"; cmp_ case_nominal "$OUT/results/case_1_1_1.jsonl" "$RES/sensitivity_runs/case_1_1_1.jsonl"
  SENS=0.8,0.8,0.8 NSEED=1 step case_failing_corner $PYTHON sens_run.py "$OUT/results/case_0.8_0.8_0.8.jsonl"; cmp_ case_failing_corner "$OUT/results/case_0.8_0.8_0.8.jsonl" "$RES/sensitivity_runs/case_0.8_0.8_0.8.jsonl"
  GRP=0.8,1,1 NSEED=1 step grp_quick $PYTHON sens_run.py "$OUT/results/grp_0.8_1_1.jsonl"; cmp_ grp_quick "$OUT/results/grp_0.8_1_1.jsonl" "$RES/group_runs/grp_0.8_1_1.jsonl"
  KT=1.2 NSEED=1 step T_quick $PYTHON sens_run.py "$OUT/results/T_1.2.jsonl"; cmp_ T_quick "$OUT/results/T_1.2.jsonl" "$RES/stress_runs/T_1.2.jsonl"
  SENS=0.8,0.8,0.8 NSEED=1 step corner_quick $PYTHON sens_corner.py "$OUT/results/corner_0.8_0.8_0.8.jsonl"; cmp_ corner_quick "$OUT/results/corner_0.8_0.8_0.8.jsonl" "$RES/stress_runs/corner_0.8_0.8_0.8.jsonl"
  KT=1.0 NSEED=1 step horizon_quick $PYTHON sens_horizon.py "$OUT/results/horizon_T_1.0.jsonl"; cmp_ horizon_quick "$OUT/results/horizon_T_1.0.jsonl" "$RES/results_h/T_1.0.jsonl"
fi

# ---- 3 extensions: A (bridge-resistance separation), B (adaptive collocation), C (ladders); seeds in the shipped files are 0..9
for r in 1 0.5 0.2 0.05; do RAB=$r step "extA_rab$r" $PYTHON extA_separation.py 0 $NS "$OUT/results/sep_rab$r.jsonl"; cmp_ "extA_rab$r" "$OUT/results/sep_rab$r.jsonl" "$RES/extA/sep_rab$r.jsonl"; done
step extB_nominal $PYTHON extB_adaptive.py 0 $NS "$OUT/results/adapt_nominal.jsonl"; cmp_ extB_nominal "$OUT/results/adapt_nominal.jsonl" "$RES/extA/adapt_nominal.jsonl"
if [ "$MODE" = full ]; then
  RAB=0.2 step extB_rab0.2 $PYTHON extB_adaptive.py 0 $NS "$OUT/results/adapt_rab0.2.jsonl"; cmp_ extB_rab0.2 "$OUT/results/adapt_rab0.2.jsonl" "$RES/extA/adapt_rab0.2.jsonl"
  KT=1.2 step extB_T14.4 $PYTHON extB_adaptive.py 0 $NS "$OUT/results/adapt_T14.4.jsonl"; cmp_ extB_T14.4 "$OUT/results/adapt_T14.4.jsonl" "$RES/extA/adapt_T14.4.jsonl"
fi
NNODE=3 MODES=lbfgs step extC_n3 $PYTHON extC_pinn.py 0 $NS 4000 "$OUT/results/ladder_n3.jsonl"; cmp_ extC_n3 "$OUT/results/ladder_n3.jsonl" "$RES/extA/ladder_n3.jsonl"
if [ "$MODE" = full ]; then
  for n in 6 10; do NNODE=$n MODES=lbfgs step extC_n$n $PYTHON extC_pinn.py 0 $NS 16000 "$OUT/results/ladder_n$n.jsonl"; cmp_ extC_n$n "$OUT/results/ladder_n$n.jsonl" "$RES/extA/ladder_n$n.jsonl"; done
fi

# ---- 4 residual enclosure (candidate procedure) and its reference check
step cert_nominal $PYTHON cert.py "$OUT/results/cert_nominal.jsonl" $SEEDQ; cmp_ cert_nominal "$OUT/results/cert_nominal.jsonl" "$RES/cert_nominal.jsonl"
if [ "$MODE" = full ]; then SENS=0.8,0.8,0.8 step cert_corner $PYTHON cert.py "$OUT/results/cert_corner.jsonl" 0 1 2; cmp_ cert_corner "$OUT/results/cert_corner.jsonl" "$RES/cert_corner.jsonl"; fi
step cert_validate $PYTHON cert_validate.py

# ---- 5 paper values: shipped results always; the new results as well in full mode
step paper_values_shipped $PYTHON check_paper_values.py "$RES"
# (the new records are compared with the shipped records one by one above; the aggregate tables checked by check_paper_values.py are built from the shipped raw files)

# ---- 6 verdict
TOT=$(( $(date +%s) - T0 ))
if [ "$MODE" = full ]; then LABEL="FULL"; else LABEL="PARTIAL (quick mode: one representative seed of each study family; NOT a full reproducibility verification)"; fi
if [ -s "$LOG/failures.txt" ]; then echo "RESULT: FAILED  [$LABEL]" | tee "$LOG/RUN_LABEL.txt"; cat "$LOG/failures.txt"; cat "$SUMMARY"; exit 1; fi
echo "RESULT: PASS  [$LABEL]  total ${TOT}s" | tee "$LOG/RUN_LABEL.txt"; cat "$SUMMARY"; exit 0
