"""Independent check of the K_d upper bounds (separate implementation, plain scipy, different grids):
 (i) K_lb: a direct high-resolution estimate of K_d (dense grids, trapezoid rule on smooth integrands); it must lie BELOW both upper bounds and close to them;
 (ii) the cell inequalities used in the upper bounds, tested on random pairs: ||P(t',s')|| <= ||P(t,s)|| exp(||M|| (|t'-t|+|s'-s|)) and the Lipschitz bound."""
import numpy as np, scipy.linalg as sla, cert, network_core as nc
T = nc.T; M = nc.Mred; mn = np.linalg.norm(M, 2)
Jz = np.zeros((5, 2)); Jz[3:5] = np.eye(2)
PT = sla.expm(M*T); Pinv = np.linalg.inv(PT[3:5, 3:5])
def P(t, s): return sla.expm(M*t) @ Jz @ Pinv @ sla.expm(M*(T - s))[3:5, :]
ns, nt = 4001, 241
ss = np.linspace(0, T, ns); tt = np.linspace(0, T, nt)
FS = np.array([sla.expm(M*(T - s))[3:5, :] for s in ss]); PU = np.array([np.linalg.norm(sla.expm(M*u), 2) for u in ss])
I2 = np.trapezoid(PU, ss)
K = 0.0
for t in tt:
    X = sla.expm(M*t) @ Jz @ Pinv; v = np.linalg.norm(np.einsum('pq,sqr->spr', X, FS), 2, axis=(1, 2)); K = max(K, np.trapezoid(v, ss))
Klb = K + 0.0     # sup_t I1 estimate
# I2(t) is nondecreasing: the true K_d >= sup_t (I1(t)+I2(t)) >= I1(t*)+I2(t*) ; use the sum at every t on the grid for the lower estimate
Klb = 0.0
for t in tt:
    X = sla.expm(M*t) @ Jz @ Pinv; v = np.linalg.norm(np.einsum('pq,sqr->spr', X, FS), 2, axis=(1, 2)); u = ss[ss <= t]
    Klb = max(Klb, np.trapezoid(v, ss) + (np.trapezoid(PU[:len(u)], u) if len(u) > 1 else 0.0))
c = cert.constants(); d = cert.constants_deriv()
print('K_lb (direct dense estimate of sup_t[I1+I2])  =', Klb)
print('K_d exponential-cell bound                    =', c['Kd'], c['Kd_parts'])
print('K_d derivative (Lipschitz) bound              =', d['Kd'], d['Kd_parts'])
print('lower estimate below both upper bounds:', Klb <= c['Kd'] and Klb <= d['Kd'])
rng = np.random.default_rng(0); worst_exp = 0; worst_lip = 0
Phimax = d['Phimax']
for _ in range(400):
    t, s = rng.uniform(0, T, 2); dt, ds = rng.uniform(-3e-5, 3e-5, 2); t2 = min(max(t + dt, 0), T); s2 = min(max(s + ds, 0), T)
    a, b = np.linalg.norm(P(t, s), 2), np.linalg.norm(P(t2, s2), 2)
    worst_exp = max(worst_exp, b/(a*np.exp(mn*(abs(t2 - t) + abs(s2 - s)))))
    lip = d['Pinv']*Phimax**2*mn*(abs(t2 - t) + abs(s2 - s)); worst_lip = max(worst_lip, (b - a)/lip)
print('cell inequality, exponential form: max of ||P(t2,s2)|| / (||P(t,s)|| e^{m(|dt|+|ds|)}) over 400 random pairs =', worst_exp, '(<=1 expected)')
print('cell inequality, Lipschitz form: max of (||P(t2,s2)||-||P(t,s)||)/(bound) =', worst_lip, '(<=1 expected)')
