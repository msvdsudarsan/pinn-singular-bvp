"""Horizon x budget study: T = 12 ms * KT (env KT), budgets N in (2000,4000,8000), Variant D/R x {L-BFGS only, Adam->L-BFGS 30%}, hard BC, 10 seeds."""
import sys, os, json
import network_pinn as npn, network_core as nc
fn = sys.argv[1]; done = set()
try:
    for l in open(fn): r = json.loads(l); done.add((r['variant'], r['mode'], r['seed'], r['N']))
except FileNotFoundError: pass
for variant in ('D', 'R'):
    for mode in ('lbfgs', 'two'): npn.build(variant, True)
for N in (2000, 4000, 8000):
    for seed in range(int(os.environ.get('NSEED', '10'))):
        for variant in ('D', 'R'):
            for mode, fr in (('lbfgs', 0.0), ('two', 0.3)):
                if (variant, mode, seed, N) in done: continue
                r = npn.run(variant, True, mode, fr, seed, N=N); r.update(N=N, T=nc.T, variant=variant, mode=mode, seed=seed)
                open(fn, 'a').write(json.dumps(r) + '\n')
                print(nc.T, N, variant, mode, seed, f"{r['relL2_total']:.2e}", f"{r['sec']:.1f}s", flush=True)
