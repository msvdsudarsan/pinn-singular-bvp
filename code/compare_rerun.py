"""Compare a re-run JSONL file with the shipped JSONL file record by record.
Usage: python3 compare_rerun.py RERUN.jsonl SHIPPED.jsonl [rtol]
Records are matched on their identity fields (variant, mode, method, seed, N/budget, rab, n, hard, frac, ex, act, width). Every numeric field except timings is compared.
Exit status 0 only if at least one record matched, no matched record differs by more than rtol (default 1e-6), and no rerun record is missing in the shipped file."""
import sys, json, math
ID = ('act', 'width', 'variant', 'mode', 'method', 'seed', 'N', 'budget', 'rab', 'n', 'hard', 'frac', 'ex', 'cfg', 'opt', 'bc')
SKIP = ('sec', 'time', 'runtime', 'wall', 'message')
def conv(v):
    if v in ('True', 'False'): return v == 'True'
    try: return int(v) if v.lstrip('-').isdigit() else float(v)
    except ValueError: return v
def load(fn):
    out = {}
    if fn.endswith('.csv'):
        import csv
        rows = [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(open(fn))]
    else:
        rows = [json.loads(l) for l in open(fn) if l.strip().startswith('{')]
    for r in rows:
        # ID fields are compared as numbers/strings so that 0.3 and 0.30 match
        out[tuple((k, repr(float(r[k])) if isinstance(r[k], (int, float)) and not isinstance(r[k], bool) else repr(r[k])) for k in ID if k in r)] = r
    return out
def flat(v, p=''):
    if isinstance(v, dict):
        for k, x in v.items(): yield from flat(x, p + '.' + k)
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v): yield from flat(x, p + '[%d]' % i)
    elif isinstance(v, bool) or v is None or isinstance(v, str): yield p, v
    else: yield p, float(v)
if __name__ == '__main__':
    a, b = load(sys.argv[1]), load(sys.argv[2]); rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
    n = 0; worst = 0.0; bad = 0; missing = 0
    for k, r in a.items():
        if k not in b: missing += 1; print('NOT IN SHIPPED:', k); continue
        n += 1; fa, fb = dict(flat(r)), dict(flat(b[k]))
        for f, x in fa.items():
            if any(s in f for s in SKIP) or f not in fb: continue
            y = fb[f]
            if isinstance(x, float) and isinstance(y, float):
                d = abs(x - y) / max(abs(y), 1e-300) if x != y else 0.0
                if math.isnan(x) and math.isnan(y): d = 0.0
                worst = max(worst, d)
                if d > rtol: bad += 1; print('DIFF', k, f, x, y)
            elif x != y: bad += 1; print('DIFF', k, f, x, y)
    ok = n > 0 and bad == 0 and missing == 0
    print(f"{'PASS' if ok else 'FAIL'}: {n} records matched, max relative difference {worst:.3e}, {bad} differing fields, {missing} unmatched; rerun={sys.argv[1]}")
    sys.exit(0 if ok else 1)
