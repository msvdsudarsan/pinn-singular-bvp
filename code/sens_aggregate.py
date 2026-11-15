import json, glob, numpy as np, csv
rows = [json.loads(l) for f in sorted(glob.glob('results/case_*.jsonl')) for l in open(f)]
val = {(r['KR'], r['KL'], r['KC']): r for r in map(json.loads, open('validation.jsonl'))}
cases = sorted({(r['KR'], r['KL'], r['KC']) for r in rows}, key=lambda c: (sum(x != 1 for x in c), c))
rng = np.random.default_rng(0)
def boot(x):                                  # bootstrap 95% CI of the median
    x = np.asarray(x); b = [np.median(rng.choice(x, len(x))) for _ in range(2000)]; return np.percentile(b, [2.5, 97.5])
def sel(c, v, m): 
    return sorted([r for r in rows if (r['KR'], r['KL'], r['KC']) == c and r['variant'] == v and r['mode'] == m], key=lambda r: r['seed'])
out = []; print(f"{len(rows)} runs, {len(cases)} cases")
hdr = ['kR','kL','kC','condM','variant','mode','n','median_relL2','IQR_lo','IQR_hi','median_relL2_v','median_relL2_i','median_relL2_alg','median_alg_violation','median_diff_res','fail_rate','median_sec']
for c in cases:
    for v in 'DR':
        for m in ('lbfgs', 'two'):
            rs = sel(c, v, m)
            if not rs: continue
            e = np.array([r['relL2_total'] for r in rs])
            out.append([*c, round(val[c]['condM'], 1), v, m, len(rs), np.median(e), *np.percentile(e, [25, 75]),
                np.median([r['relL2_v'] for r in rs]), np.median([r['relL2_i'] for r in rs]), np.median([r['relL2_alg'] for r in rs]),
                np.median([r['alg_constraint_violation_max'] for r in rs]), np.median([r['diff_res_rms'] for r in rs]),
                np.mean([r['fail'] for r in rs]), np.median([r['sec'] for r in rs])])
with open('sensitivity_table.csv', 'w', newline='') as f: w = csv.writer(f); w.writerow(hdr); w.writerows(out)
# paired comparisons per case (log10 ratio, seed-paired): D/R (same mode) and two/lbfgs (same variant)
comp = []
for c in cases:
    d = {}
    for name, (v1, m1, v2, m2) in dict(D_over_R_lbfgs=('D','lbfgs','R','lbfgs'), D_over_R_two=('D','two','R','two'),
                                      two_over_lbfgs_R=('R','two','R','lbfgs'), two_over_lbfgs_D=('D','two','D','lbfgs')).items():
        a, b = sel(c, v1, m1), sel(c, v2, m2)
        if len(a) == len(b) and a:
            lr = np.log10([x['relL2_total']/y['relL2_total'] for x, y in zip(a, b)])
            wins = int(np.sum(lr > 0)); d[name] = (float(np.median(lr)), *map(float, boot(lr)), wins, len(lr))
    comp.append((c, d))
with open('sensitivity_paired.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['kR','kL','kC','comparison','median_log10_ratio','ci_lo','ci_hi','n_ratio_gt_1','n'])
    for c, d in comp:
        for k, v in d.items(): w.writerow([*c, k, *v])
# one-at-a-time sensitivity of the best configuration (R, L-BFGS-only) relative to nominal
nom = np.median([r['relL2_total'] for r in sel((1.0,1.0,1.0), 'R', 'lbfgs')])
print('nominal R/lbfgs median', nom)
for c in cases:
    rs = sel(c, 'R', 'lbfgs'); 
    if rs: print(c, f"R/lbfgs med={np.median([r['relL2_total'] for r in rs]):.2e}  D/lbfgs med={np.median([r['relL2_total'] for r in sel(c,'D','lbfgs')]):.2e}  condM={val[c]['condM']:.1f}")
tot = {}
for c, d in comp:
    for k, v in d.items(): tot.setdefault(k, []).append(v)
for k, vs in tot.items():
    print(k, 'cases where median log-ratio>0:', sum(v[0] > 0 for v in vs), '/', len(vs), '; CI excludes 0 (>0):', sum(v[1] > 0 for v in vs), '; (<0):', sum(v[2] < 0 for v in vs))
print('failures (>5% error):', sum(r['fail'] for r in rows), 'of', len(rows))
print('max rel error overall:', max(r['relL2_total'] for r in rows))
# figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
lab = [f"{c[0]:.1f},{c[1]:.1f},{c[2]:.1f}" for c in cases]
for v, mk in (('D', 'o'), ('R', 's')):
    for m, ls in (('lbfgs', '-'), ('two', '--')):
        y = [np.median([r['relL2_total'] for r in sel(c, v, m)]) for c in cases]
        ax[0].semilogy(range(len(cases)), y, marker=mk, ls=ls, label=f"Variant {v}, {'L-BFGS only' if m=='lbfgs' else 'Adam→L-BFGS'}")
ax[0].set_xticks(range(len(cases))); ax[0].set_xticklabels(lab, rotation=75, fontsize=7); ax[0].set_xlabel('(kR, kL, kC)'); ax[0].set_ylabel('median relative L2 error (10 seeds)'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].semilogy([val[c]['condM'] for c in cases], [np.median([r['relL2_total'] for r in sel(c, 'R', 'lbfgs')]) for c in cases], 's')
ax[1].set_xlabel('condition number of boundary operator'); ax[1].set_ylabel('median error, Variant R, L-BFGS only'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig('fig_sensitivity.png', dpi=200)
