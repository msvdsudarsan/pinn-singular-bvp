"""Vector (PDF) versions of the three data figures, drawn from the raw records in ../results. Usage: python3 make_figures.py [outdir]"""
import json, glob, os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42; matplotlib.rcParams['font.size'] = 7.5
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
# --- Adam-fraction sweep (from the summary table of the 560 sweep runs)
import csv
S = list(csv.DictReader(open(os.path.join(RES, 'switch_sweep_summary.csv'))))
fig, axs = plt.subplots(2, 4, figsize=(6.8, 4.3), sharex=True)
for row, bc in enumerate(('soft', 'hard')):
    for col in range(4):
        ex = str(col + 1); a = axs[row, col]
        d = sorted([r for r in S if r['ex'] == ex and r['bc'] == bc], key=lambda r: float(r['adam_frac']))
        x = [float(r['adam_frac']) for r in d]; m = [float(r['median']) for r in d]
        a.semilogy(x, m, 'o-', color='tab:blue', ms=2.8, lw=1.0); a.fill_between(x, [float(r['q25']) for r in d], [float(r['q75']) for r in d], alpha=.25, color='tab:blue')
        a.axhline(0.05, color='red', ls=':', lw=1.2); a.grid(alpha=.3, which='both'); a.set_title(f'Example {ex}, ' + ('penalty' if bc == 'soft' else 'exact') + ' BC', fontsize=7.5)
        if col == 0: a.set_ylabel('Relative $L_2$ error', fontsize=7.5)
        if row == 1: a.set_xlabel('Adam fraction $\\varphi$', fontsize=7.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_switch_sweep.pdf'))
# --- sensitivity
rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(RES, 'sensitivity_runs', 'case_*.jsonl'))) for l in open(f)]
val = {(r['KR'], r['KL'], r['KC']): r for r in map(json.loads, open(os.path.join(RES, 'validation.jsonl')))}
cases = sorted({(r['KR'], r['KL'], r['KC']) for r in rows}, key=lambda c: (sum(x != 1 for x in c), c))
def sel(c, v, m): return [r for r in rows if (r['KR'], r['KL'], r['KC']) == c and r['variant'] == v and r['mode'] == m]
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
lab = [f"{c[0]:.1f},{c[1]:.1f},{c[2]:.1f}" for c in cases]
for v, mk in (('D', 'o'), ('R', 's')):
    for m, ls in (('lbfgs', '-'), ('two', '--')):
        y = [np.median([r['relL2_total'] for r in sel(c, v, m)]) for c in cases]
        ax[0].semilogy(range(len(cases)), y, marker=mk, ls=ls, label=f"Variant {v}, {'L-BFGS only' if m == 'lbfgs' else 'Adam→L-BFGS'}")
ax[0].set_xticks(range(len(cases))); ax[0].set_xticklabels(lab, rotation=75, fontsize=6); ax[0].set_xlabel(r'$(k_R,\,k_L,\,k_C)$'); ax[0].set_ylabel('median relative $L_2$ error (10 seeds)'); ax[0].legend(fontsize=6.5); ax[0].grid(alpha=.3)
ax[1].semilogy([val[c]['condM'] for c in cases], [np.median([r['relL2_total'] for r in sel(c, 'R', 'lbfgs')]) for c in cases], 's')
ax[1].set_xlabel('condition number of boundary operator'); ax[1].set_ylabel('median error, Variant R, L-BFGS only'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_sensitivity.pdf'))
print('wrote 2 figures to', OUT)
