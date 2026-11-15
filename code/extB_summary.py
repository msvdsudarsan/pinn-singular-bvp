import json,glob,sys,numpy as np
for f in sorted(glob.glob(sys.argv[1]+'/adapt_*.jsonl')):
    R=[json.loads(l) for l in open(f)]; print(f.split('/')[-1],len(R))
    for v in 'DR':
        for m in ('fixed','adapt','adaptw'):
            a=[r for r in R if r['variant']==v and r['method']==m]
            if a: print(' ',v,m,len(a),'median %.2e'%np.median([r['relL2'] for r in a]),'IQR %.1e-%.1e'%tuple(np.percentile([r['relL2'] for r in a],[25,75])),'fail',sum(r['fail'] for r in a),'sec %.0f'%np.mean([r['sec'] for r in a]))
