import json, glob, csv, numpy as np
rng = np.random.default_rng(2)
L = lambda f: [json.loads(l) for l in open(f)]
def boot(x): return np.percentile([np.median(rng.choice(x, len(x))) for _ in range(2000)], [2.5, 97.5])
def pr(a, b):
    lr = np.log10([x['relL2_total']/y['relL2_total'] for x, y in zip(a, b)]); lo, hi = boot(lr); return f"{10**np.median(lr):.2f} [{10**lo:.2f}, {10**hi:.2f}]"
val = {}
for f in ('validation_rab.jsonl', 'validation_T.jsonl'):
    for l in open(f): r = json.loads(l); val[(r['RAB'], round(r['T'], 5))] = r
out = [['study','setting','condA22','condBC','R_lbfgs','R_two','D_lbfgs','D_two','failed_runs_of_40','D/R_lbfgs [95% CI]','D/R_two [95% CI]','two/lbfgs_R [95% CI]','two/lbfgs_D [95% CI]']]
for study, fs in (('Rab only', ['rab_0.2', 'rab_0.1', 'rab_0.05']), ('horizon T', ['T_0.8', 'T_1.2'])):
    for f in fs:
        rs = L(f'results_x/{f}.jsonl'); k = (rs[0]['RAB'], round(rs[0]['T'], 5)); sel = lambda v, m: sorted([r for r in rs if r['variant'] == v and r['mode'] == m], key=lambda r: r['seed'])
        med = lambda v, m: f"{np.median([r['relL2_total'] for r in sel(v, m)]):.2e}"
        out.append([study, f"Rab x{rs[0]['RAB']}" if study == 'Rab only' else f"T = {rs[0]['T']*1000:.1f} ms", round(val[k]['condA22'], 2), round(val[k]['condM'], 1), med('R','lbfgs'), med('R','two'), med('D','lbfgs'), med('D','two'), sum(r['fail'] for r in rs),
            pr(sel('D','lbfgs'), sel('R','lbfgs')), pr(sel('D','two'), sel('R','two')), pr(sel('R','two'), sel('R','lbfgs')), pr(sel('D','two'), sel('D','lbfgs'))])
csv.writer(open('stress_horizon_table.csv', 'w', newline='')).writerows(out)
for r in out: print(r)
cor = [['corner','variant','median_err','min','max','n_above_2pct','n_above_5pct','n']]
for f in sorted(glob.glob('results_x/corner_*.jsonl')):
    rs = L(f)
    for v in 'RD':
        e = np.array([r['relL2_total'] for r in rs if r['variant'] == v]); cor.append([f.split('corner_')[1][:-6], v, f"{np.median(e):.2e}", f"{e.min():.1e}", f"{e.max():.1e}", int((e > .02).sum()), int((e > .05).sum()), len(e)])
csv.writer(open('corner_replication_table.csv', 'w', newline='')).writerows(cor)
for r in cor: print(r)
