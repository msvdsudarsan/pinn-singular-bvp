# Code, data and results: PINNs for two-point boundary value problems of singular matrix differential equations

This repository contains the code, the raw per-run records and the summary tables behind the numerical studies. The manuscript source and PDF are not included.

## Contents
- `code/original_examples_*.py` - the four benchmark problems: independent reference solutions, PINN training (soft and exact boundary conditions), Adam-fraction sweep and activation/width ablation.
- `code/network_core.py`, `code/network_pinn.py` - the coupled three-node RLC descriptor network (matrices, references, full-descriptor and reduced PINN).
- `code/sens_run.py`, `sens_corner.py`, `sens_horizon.py`, `run_*.sh`, `*_aggregate.py` - sensitivity, stress, horizon and budget studies and their aggregation.
- `code/cert.py`, `cert_*.py` - the conservative residual-enclosure procedure and its checks.
- `code/run_all.sh [quick|full]` - master script for every study (see Reproducing). `code/run_grp.sh`, `run_x.sh`, `run_horizon.sh` and `common.sh` are older partial runners kept for single studies. All scripts write to `rerun_output/` (or `OUT=`), never overwrite the shipped raw results, record failed steps and exit non-zero if any step failed.
- `code/compare_rerun.py` - compares a re-run JSONL/CSV with the shipped records, record by record (matched on variant, mode, seed, budget and the other identity fields; timings are ignored).
- `code/check_paper_values.py` - checks the numbers quoted in the manuscript against the shipped result files (108 checks).
- `code/make_figures.py`, `code/draw_topology.py` - vector (PDF) versions of the figures, drawn from the raw records in `results/`.
- `code/mna_validate.py` - independent, topology-based modified nodal analysis (MNA) reference; it is not a SPICE simulation.
- `code/proof_verification.py` - numerical, symbolic and sampled checks of the statements in the supplement (38 checks of different kinds, not 38 independent proofs). It reads the supplement's LaTeX source, which is not distributed here: pass its path as an argument (`python3 proof_verification.py /path/to/supplement.tex`) or set `SUPP=` for `run_all.sh`.
- `results/` - raw JSONL/CSV records of every run and the summary tables.

## Reproducing
Requirements: Python 3.10+, JAX (float64), NumPy, SciPy, mpmath, SymPy, pandas. Versions used for the shipped runs: Python 3.13, JAX 0.12.0.1, NumPy 2.5.3, SciPy 1.18.1, mpmath 1.3.0, SymPy 1.14.0, pandas 3.0.5 (CPU, two cores). Seeds are the loop indices 0-9 (JAX `PRNGKey(seed)`); on the same machine records are reproduced bit for bit, on other machines small floating-point differences can grow along an optimisation path, so compare statistics (medians, failure counts) if `compare_rerun.py` reports differences.

```
cd code
SUPP=/path/to/supplement.tex ./run_all.sh quick     # about 40 minutes (2429 s in the logged run): one representative seed (two for the extension and enclosure studies) of each study family, compared with the shipped records. A subset of the configurations and seeds; labelled PARTIAL.
SUPP=/path/to/supplement.tex ./run_all.sh full      # all studies and seeds: about 8.6 hours of summed step time on two CPU cores (30,797 s in the logged run). Labelled FULL.
```
Quick mode covers: the four original examples, the switch-point sweep, the activation/width ablation, the RLC nominal, failing-corner, group, horizon-scaling and bridge-resistance (extension A) cases, adaptive collocation, the 3-node ladder, the enclosure computation and its checks, proof verification, nodal-analysis validation and paper-value checks. It does not cover the remaining seeds, the other 14 sensitivity cases, the other group scalings, the 6- and 10-node ladders or the stressed-case adaptive and enclosure runs.

Both modes: log the environment (`rerun_output/logs/environment.txt`), the commands (`commands.txt`), the exit code and run time of every step (`summary.tsv`), compare every re-run file with the shipped file (`compare_*.log`) and finish with `check_paper_values.py` on the shipped results. The result and its label (`FULL` or `PARTIAL`) are printed and written to `rerun_output/logs/RUN_LABEL.txt`; the exit status is 0 only if every step passed. The logged full run (`results/rerun_full_log/`: RESULT PASS [FULL], 100 steps exit 0, 47 comparisons matched with maximum relative difference 0) was split into parallel processes because of a two-hour limit on background commands: `ONLY_RE` and `SKIP_RE` (regular expressions on step names) select steps, and a step with a done marker in `rerun_output/logs/done/` is skipped, so an interrupted run can be resumed; a final plain `./run_all.sh full` then prints the verdict. Use a fresh `OUT=` directory for a clean run; existing records in an output directory are resumed or appended to, with a warning. `full` clears `code/log_sweep.txt` so that the switch-point sweep cannot skip runs silently.

Single studies (environment variables select the case): `SENS=0.8,1.0,1.2 python3 sens_run.py out.jsonl` (resistance, inductance, capacitance scaling), `GRP=` (resistance groups), `KT=1.2` (horizon scaling), `RAB=0.2` (bridge resistance), `NSEED=1` (number of seeds, default 10).

Not covered by `run_all.sh`: the aggregation tables (`*_aggregate.py`) and the figures (`make_figures.py`, `draw_topology.py`) are built from the shipped raw records; run them separately. The bridge-resistance runs of `sens_run.py` (`rab_*`) are run in `full` mode but have no shipped file of the same name to compare with; their numbers enter through the extension studies (`results/extA/`).

## Notes
Runs are single-machine, CPU, double precision; wall-clock times depend on the machine. The failure threshold is a relative L2 error of 5%.

## Extension experiments (error decomposition, adaptive collocation, ladder networks)
Code: `code/extA_separation.py`, `code/extA_landscape.py`, `code/extB_adaptive.py`, `code/extB_audit.py`, `code/extB_summary.py`, `code/ladder_core.py`, `code/extC_pinn.py`. Raw records and logs: `results/extA/`.
```
cd code
RAB=1 python3 extA_separation.py 0 10 ../results/extA/sep_rab1.jsonl          # also RAB=0.5, 0.2, 0.05
RAB=1 python3 extB_adaptive.py 0 10 ../results/extA/adapt_nominal.jsonl      # also RAB=0.2 and KT=1.2
python3 extB_audit.py ../results/extA                                         # paired seed-wise comparison
python3 ladder_core.py                                                        # reference check, NNODE=3|6|10
NNODE=6 MODES=lbfgs python3 extC_pinn.py 0 10 16000 ../results/extA/ladder_n6.jsonl
```
Limits: the operator norms in extA are obtained from a collocation discretization (not rigorous); the ladder study uses 10 seeds (records include actual nfev and optimizer status; the earlier 5-seed records are in results/extA/old5); adaptive collocation did not improve on uniform collocation.
