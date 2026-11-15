"""Paired seed-wise audit of Extension B: log10 ratio (adaptive/fixed) with bootstrap 95% interval; same seed => same initialisation, same network, same budget."""
import json,glob,sys,numpy as np
rng=np.random.default_rng(0)
for f in sorted(glob.glob(sys.argv[1]+'/adapt_*.jsonl')):
    R=[json.loads(l) for l in open(f)]; g=lambda v,m:{r['seed']:r for r in R if r['variant']==v and r['method']==m}
    print(f.split('/')[-1])
    for v,m in (('D','adapt'),('D','adaptw'),('R','adapt')):
        a,b=g(v,m),g(v,'fixed'); s=sorted(set(a)&set(b)); d=np.array([np.log10(a[k]['relL2']/b[k]['relL2']) for k in s])
        bs=[np.median(rng.choice(d,len(d))) for _ in range(5000)]
        print(f'  {v} {m} vs fixed: n={len(s)} median ratio {10**np.median(d):.2f} CI [{10**np.percentile(bs,2.5):.2f},{10**np.percentile(bs,97.5):.2f}] adaptive better in {(d<0).sum()}/{len(d)} seeds; max |nfev difference|: {max(abs(a[k]["nfev"]-b[k]["nfev"]) for k in s)} (nominal budget 2000)')
