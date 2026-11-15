"""Consistency check: numbers quoted in the paper against the raw per-run records.

Usage:  python3 check_paper_values.py [RESULTS_DIR]      (default: ../results, the shipped raw records)
Each line prints PASS or FAIL with the value recomputed from the records and the value quoted in the paper.
Exit status 0 only if every check passes. Numbers are compared after rounding to the precision used in the paper.
"""
import sys, os, json, glob
import numpy as np, pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
fails = []
n_checks = 0
def check(name, got, want, tol=0.0):
    global n_checks
    n_checks += 1
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float, np.floating)) else (got == want)
    print(('PASS ' if ok else 'FAIL ') + name + f': recomputed {got!r}, paper {want!r}')
    if not ok: fails.append(name)
def sig2(x):
    return float(f'{x:.1e}')
def rd(path): return pd.read_csv(os.path.join(R, path))
def rj(path): return [json.loads(l) for l in open(os.path.join(R, path)) if l.strip()]

# 1. Table 2: median relative L2 error and failed runs, four benchmark problems (paper values, two significant digits)
T2 = {  # (example, bc): [(L-BFGS only), (30% Adam), (40% Adam), (Adam only)] as (median, failures)
 (1,'hard'): [(2.9e-5,0),(7.3e-5,0),(8.9e-5,0),(6.7e-3,0)], (1,'soft'): [(6.8e-5,0),(1.4e-4,0),(1.8e-4,0),(7.4e-3,0)],
 (2,'hard'): [(2.2e-3,0),(1.1e-3,0),(6.0e-4,0),(3.0e-2,1)], (2,'soft'): [(5.9e-3,0),(9.6e-3,0),(1.4e-2,0),(8.5e-1,10)],
 (3,'hard'): [(6.3e-5,0),(1.3e-4,0),(1.1e-4,0),(2.8e-3,0)], (3,'soft'): [(8.7e-4,0),(2.2e-3,0),(1.9e-3,0),(1.6e-1,10)],
 (4,'hard'): [(2.8e-4,0),(2.2e-4,0),(3.1e-4,0),(1.2e-2,0)], (4,'soft'): [(1.6e-3,0),(1.9e-3,0),(2.4e-3,0),(1.9e-1,10)]}
sw = rd('switch_sweep_summary.csv')
for (ex, bc), cols in T2.items():
    for fr, (m, f) in zip((0.0, 0.3, 0.4, 1.0), cols):
        row = sw[(sw.ex == ex) & (sw.bc == bc) & (np.isclose(sw.adam_frac, fr))]
        got_m = sig2(float(row['median'].iloc[0])); got_f = int(row['fails'].iloc[0])
        check(f'Table 2 ex{ex} {bc} frac {fr} median', got_m, m, 1e-12 + 1e-9*abs(m))
        check(f'Table 2 ex{ex} {bc} frac {fr} failures', got_f, f)

# 2. Nominal network, full (D) versus reduced (R)
st = rd('sensitivity_table.csv'); sp = rd('sensitivity_paired.csv')
nom = st[(st.kR == 1) & (st.kL == 1) & (st.kC == 1)].set_index(['variant', 'mode']).median_relL2
for key, want in {('D','lbfgs'):1.7e-3, ('R','lbfgs'):4.4e-4, ('D','two'):3.1e-3, ('R','two'):5.9e-4}.items():
    check(f'nominal median {key}', sig2(nom[key]), want, 1e-9*want)
pn = sp[(sp.kR == 1) & (sp.kL == 1) & (sp.kC == 1)].set_index('comparison').median_log10_ratio
check('nominal paired D/R (L-BFGS)', round(10**pn['D_over_R_lbfgs'], 1), 3.3, 1e-9)
check('nominal paired D/R (two-stage)', round(10**pn['D_over_R_two'], 1), 5.0, 1e-9)

pv = rd('network_D_vs_R_paired.csv'); pl = pv[(pv.bc == 'hard') & (pv.optimizer == 'lbfgs')].iloc[0]
check('median time per run, D (L-BFGS only), s', round(float(pl.time_D), 1), 10.4, 1e-9); check('median time per run, R (L-BFGS only), s', round(float(pl.time_R), 1), 10.1, 1e-9)

# 3. 15 parameter combinations: R below D everywhere, ratio of medians, corner failures
g = st.groupby(['kR', 'kL', 'kC'])
check('number of parameter combinations', len(g), 15)
allbelow = True; ratios = []
for _, d in g:
    t = d.set_index(['variant', 'mode']).median_relL2
    allbelow &= bool(t[('R','lbfgs')] < t[('D','lbfgs')]) and bool(t[('R','two')] < t[('D','two')])
    ratios.append(t[('R','lbfgs')] / t[('D','lbfgs')])
check('R lower median than D in all 15 combinations, both optimizers', allbelow, True)
check('smallest R/D ratio of medians (L-BFGS)', round(min(ratios), 2), 0.11, 1e-9)
check('largest R/D ratio of medians (L-BFGS)', round(max(ratios), 2), 0.66, 1e-9)
c1 = st[(st.kR == 0.8) & (st.kL == 0.8) & (st.kC == 0.8)]
check('corner (0.8,0.8,0.8): failed runs of 40', int(round((c1.fail_rate * c1.n).sum())), 40)
c2 = st[(st.kR == 1.2) & (st.kL == 0.8) & (st.kC == 0.8)]
check('corner (1.2,0.8,0.8): D failures of 20', int(round((c2[c2.variant == 'D'].fail_rate * c2[c2.variant == 'D'].n).sum())), 20)
check('corner (1.2,0.8,0.8): R failures of 20', int(round((c2[c2.variant == 'R'].fail_rate * c2[c2.variant == 'R'].n).sum())), 14)

# 4. Bridge-resistance stress test: paired D/R ratio (L-BFGS)
sh = rd('stress_horizon_table.csv').set_index('setting')
for s, want in {'Rab x0.2': 16, 'Rab x0.1': 39, 'Rab x0.05': 122}.items():
    check(f'stress {s} paired D/R', round(float(sh.loc[s, 'D/R_lbfgs [95% CI]'].split()[0])), want)
check('condition number of A22 at 5% of nominal', round(float(sh.loc['Rab x0.05', 'condA22']), 1), 17.3, 1e-9)

# 5. Horizon and budget
h = rd('horizon_summary.csv'); h['T'] = (h.T_s * 1000).round(1)
f = h.groupby(['T', 'N']).fails_gt5pct.sum()
check('T=9.6 ms failures (all budgets)', int(f[9.6].sum()), 0); check('T=12 ms failures (all budgets)', int(f[12.0].sum()), 0)
check('T=14.4 ms, 2000 evaluations: failures of 40', int(f[(14.4, 2000)]), 40)
check('T=14.4 ms, 8000 evaluations: failures of 40', int(f[(14.4, 8000)]), 6)
check('T=14.4 ms, 8000 evaluations: reduced-form failures of 20', int(h[(h['T'] == 14.4) & (h.N == 8000) & (h.variant == 'R')].fails_gt5pct.sum()), 1)

# 6. Residual-enclosure bound (candidate procedure): ratio bound / sampled supremum error, L-BFGS only
cn = rj('cert_nominal.jsonl'); cc = rj('cert_corner.jsonl')
rn = np.concatenate([x['ratio'] for x in cn]); rc = np.concatenate([x['ratio'] for x in cc])
check('enclosure ratio, nominal: minimum', round(rn.min()), 39); check('enclosure ratio, nominal: maximum', round(rn.max()), 333)
check('enclosure ratio, failing case: minimum', round(rc.min(), 1), 4.1, 1e-9); check('enclosure ratio, failing case: maximum', round(rc.max()), 46)
check('enclosure columns tested', len(rn) + len(rc), 48); check('bound above sampled error in all columns', bool((np.concatenate([rn, rc]) > 1).all()), True)
for v, (lo, med, hi) in {'R': (2.2, 4.0, 5.3), 'D': (2.6, 3.2, 5.2)}.items():
    q = np.concatenate([np.array(x['sup_rd']) / np.array(x['train_rms_residual']) for x in cn if x['variant'] == v])
    check(f'sup / rms residual at nominal, variant {v}: min', round(q.min(), 1), lo, 1e-9)
    check(f'sup / rms residual at nominal, variant {v}: median', round(float(np.median(q)), 1), med, 1e-9)
    check(f'sup / rms residual at nominal, variant {v}: max', round(q.max(), 1), hi, 1e-9)
# 7. Mechanism study: 80 networks
sep = [x for fn in sorted(glob.glob(os.path.join(R, 'extA', 'sep_rab*.jsonl'))) for x in map(json.loads, open(fn)) ]
check('networks in the mechanism study', len(sep), 80); check('bound holds for all networks', sum(bool(x['holds']) for x in sep), 80)
rr = [x['bound'] / x['sup_err'] for x in sep]
check('bound / sampled error: minimum', round(min(rr), 1), 2.9, 1e-9); check('bound / sampled error: maximum', round(max(rr)), 140)
def med_r1(var, rab): return float(np.median([x['sup_r1'] for x in sep if x['variant'] == var and np.isclose(x['rab'], rab)]))
check('differential residual of D, nominal', sig2(med_r1('D', 1.0)), 2.6e-2, 1e-9); check('differential residual of D, 5% bridge resistance', sig2(med_r1('D', 0.05)), 2.1e-1, 1e-9)
rrv = [med_r1('R', r) for r in (1.0, 0.5, 0.2, 0.05)]
check('differential residual of R, smallest median', sig2(min(rrv)), 9.5e-3, 1e-9); check('differential residual of R, largest median', sig2(max(rrv)), 1.7e-2, 1e-9)

print(f'\n{n_checks - len(fails)} of {n_checks} checks PASS')
if fails: print('FAILED:', *fails, sep='\n  '); sys.exit(1)
