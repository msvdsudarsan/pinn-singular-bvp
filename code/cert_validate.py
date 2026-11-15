"""Validation of the float interval enclosure of the network (value, 1st and 2nd tau-derivative) against 50-digit exact evaluations (mpmath):
(a) rounding: |float centre - exact| <= radius at zero width (margin adequacy), reported as the largest ratio error/radius;
(b) range: exact values at random interior points of a subinterval lie inside the enclosure."""
import numpy as np, mpmath as mp, sys
import cert
mp.mp.dps = 50
unr, flat, evaluate = cert.train('R', 0)
ps = [(np.asarray(W), np.asarray(b)) for W, b in unr(flat)]
Wm = [(mp.matrix(W.tolist()), mp.matrix(b.tolist())) for W, b in ps]
def exact(tau):
    h = mp.matrix([mp.mpf(tau)]); h1 = mp.matrix([1]); h2 = mp.matrix([0])
    for W, b in Wm[:-1]:
        z = W*h + b; z1 = W*h1; z2 = W*h2
        a = z.copy(); 
        for i in range(a.rows): a[i] = mp.tanh(z[i])
        s = a.copy()
        for i in range(a.rows): s[i] = 1 - a[i]**2
        a1 = z1.copy(); a2 = z2.copy()
        for i in range(a.rows):
            a1[i] = s[i]*z1[i]; a2[i] = s[i]*z2[i] - 2*a[i]*s[i]*z1[i]**2
        h, h1, h2 = a, a1, a2
    W, b = Wm[-1]
    return [np.array((W*q + (b if k == 0 else 0)).tolist(), dtype=object).reshape(-1) for k, q in enumerate((h, h1, h2))]
nout = 5
rng = np.random.default_rng(1)
# (a) rounding at 30 points
pts = np.linspace(0.0, 1.0, 30); worst = 0.0
fo = cert.forward(ps, pts, np.zeros_like(pts), nout)
for k, t in enumerate(pts):
    ex = exact(t)
    for d in range(3):
        m = fo[d][0][k].reshape(-1); r = fo[d][1][k].reshape(-1)
        err = np.array([abs(mp.mpf(m[i]) - ex[d][i]) for i in range(len(m))], dtype=float)
        worst = max(worst, float((err/r).max()))
print('(a) largest |float centre - exact| / radius over 30 points, 3 derivative orders, all outputs:', worst)
# (b) containment over random interior points
K = 2000; edges = np.linspace(0, 1, K + 1); cs = 0.5*(edges[1:] + edges[:-1]); ws = 0.5*np.diff(edges)
idx = [0, 400, 1000, 1500, 1999]; viol = 0; tot = 0; slack = []
for k in idx:
    fo = cert.forward(ps, np.array([cs[k]]), np.array([ws[k]]), nout)
    for t in cs[k] + ws[k]*rng.uniform(-1, 1, 12):
        ex = exact(t)
        for d in range(3):
            m = fo[d][0][0].reshape(-1); r = fo[d][1][0].reshape(-1)
            for i in range(len(m)):
                tot += 1; dev = abs(ex[d][i] - mp.mpf(m[i]))
                if dev > r[i]: viol += 1
                slack.append(float(dev/r[i]))
print('(b) containment violations:', viol, 'of', tot, '; largest deviation/radius:', max(slack))
