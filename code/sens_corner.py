"""10-seed higher-budget replication (8001 evaluations, L-BFGS only, hard BC) for the two failing corners; SENS env gives the case."""
import sys, os, json, network_pinn as npn, network_core as nc
fn = sys.argv[1]
for v in 'RD': npn.build(v, True)
for seed in range(int(os.environ.get('NSEED', '10'))):
    for v in 'RD':
        r = npn.run(v, True, 'lbfgs', 0.0, seed, N=8000); r.update(KR=nc.KR, KL=nc.KL, KC=nc.KC, N=8000)
        open(fn, 'a').write(json.dumps(r) + '\n'); print(nc.KR, nc.KL, nc.KC, v, seed, f"{r['relL2_total']:.2e}", flush=True)
