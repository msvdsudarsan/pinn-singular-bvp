"""One (kR,kL,kC) case (given by env SENS): Variant D/R x {L-BFGS-only, Adam->L-BFGS (30% Adam)}, hard BC, 10 seeds, N=2000 evaluations."""
import sys, os, json
import network_pinn as npn, network_core as nc
fn = sys.argv[1]; done = set()
try:
    for l in open(fn): r = json.loads(l); done.add((r['variant'], r['mode'], r['seed']))
except FileNotFoundError: pass
for variant in ('D', 'R'):
    for mode in ('lbfgs', 'two'): npn.build(variant, True)
for seed in range(int(os.environ.get('NSEED', '10'))):
    for variant in ('D', 'R'):
        for mode, fr in (('lbfgs', 0.0), ('two', 0.3)):
            if (variant, mode, seed) in done: continue
            r = npn.run(variant, True, mode, fr, seed, N=2000); r.update(KR=nc.KR, KL=nc.KL, KC=nc.KC, KW=nc.KW, KBR=nc.KBR, KSH=nc.KSH, RAB=nc.RAB, T=nc.T)
            open(fn, 'a').write(json.dumps(r) + '\n')
            print(nc.KW, nc.KBR, nc.KSH, variant, mode, seed, f"{r['relL2_total']:.2e}", f"{r['sec']:.1f}s", flush=True)
