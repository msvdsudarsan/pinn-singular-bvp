import json, glob, csv, numpy as np
rng = np.random.default_rng(1)
rows = [json.loads(l) for f in sorted(glob.glob('results_grp/*.jsonl')) for l in open(f)]
val = {(r['KW'], r['KBR'], r['KSH']): r for r in map(json.loads, open('validation_grp.jsonl'))}
key = lambda r: (r['KW'], r['KBR'], r['KSH'])
cases = [(0.8,1,1),(1.2,1,1),(1,0.8,1),(1,1.2,1),(1,1,0.8),(1,1,1.2)]
sel = lambda c, v, m: sorted([r for r in rows if key(r) == c and r['variant'] == v and r['mode'] == m], key=lambda r: r['seed'])
def boot(x): return np.percentile([np.median(rng.choice(x, len(x))) for _ in range(2000)], [2.5, 97.5])
print(len(rows), 'runs; failures', sum(r['fail'] for r in rows), 'max err', max(r['relL2_total'] for r in rows))
out = [['kW','kBr','kSh','condA22','condM','R_lbfgs','R_two','D_lbfgs','D_two','failed_runs','D/R_lbfgs (ratio, CI lo, CI hi)','D/R_two','two/lbfgs_R','two/lbfgs_D']]
for c in cases:
    med = {(v, m): np.median([r['relL2_total'] for r in sel(c, v, m)]) for v in 'RD' for m in ('lbfgs', 'two')}
    nf = sum(r['fail'] for r in rows if key(r) == c)
    def pr(a, b):
        lr = np.log10([x['relL2_total']/y['relL2_total'] for x, y in zip(a, b)]); lo, hi = boot(lr)
        return f"{10**np.median(lr):.2f} [{10**lo:.2f}, {10**hi:.2f}]"
    out.append([*c, round(val[c]['condA22'], 3), round(val[c]['condM'], 1), *(f"{med[k]:.2e}" for k in (('R','lbfgs'),('R','two'),('D','lbfgs'),('D','two'))), nf,
        pr(sel(c,'D','lbfgs'), sel(c,'R','lbfgs')), pr(sel(c,'D','two'), sel(c,'R','two')), pr(sel(c,'R','two'), sel(c,'R','lbfgs')), pr(sel(c,'D','two'), sel(c,'D','lbfgs'))])
csv.writer(open('group_sensitivity_table.csv', 'w', newline='')).writerows(out)
for r in out: print(r)
