"""Independent, topology-based Modified Nodal Analysis (MNA) reference for the coupled three-node RLC network.
The netlist below is typed from the circuit description (element values and connections); NO hand-derived descriptor matrix is used.
This is NOT a SPICE simulation. Hand-derived references (network_core) are used only for comparison.
Usage: SENS=kR,kL,kC python3 mna_validate.py"""
import os, json, numpy as np
import network_core as nc            # hand-derived matrices/reference (for comparison) and the problem data (sources, boundary data, T)
kR, kL, kC = (float(v) for v in os.environ.get('SENS', '1,1,1').split(','))
# ---------------- netlist: (kind, name, node+, node-, value[, series R]) ; ground = '0' ----------------
NET = [('C', 'C1', '1', '0', 1.0e-3), ('C', 'C2', '2', '0', 0.8e-3), ('C', 'C3', '3', '0', 1.2e-3),
       ('R', 'Rs', '1', '0', 10.0), ('R', 'R2', '2', '0', 20.0), ('R', 'Rt', '3', '0', 5.0),
       ('R', 'R1a', '1', 'a', 5.0), ('R', 'Rab', 'a', 'b', 10.0), ('R', 'Rb3', 'b', '3', 5.0),
       ('R', 'Ra0', 'a', '0', 20.0), ('R', 'Rb0', 'b', '0', 25.0),
       ('RL', 'L1', '1', '2', 10e-3, 0.5), ('RL', 'L2', '2', '3', 15e-3, 0.8),         # inductor with series winding resistance, branch current unknown
       ('I', 'I1', '0', '1', 0), ('I', 'I3', '0', '3', 1)]                               # current sources injecting into node 1 and node 3 (input index 0, 1)
def scale(el):
    k, nm, a, b, v, *rest = el
    if k == 'R': return (k, nm, a, b, v*kR)
    if k == 'C': return (k, nm, a, b, v*kC)
    if k == 'RL': return (k, nm, a, b, v*kL, rest[0]*kR)
    return el
NET = [scale(e) for e in NET]
node_names = ['1', '2', '3', 'a', 'b']; nidx = {n: i for i, n in enumerate(node_names)}
branches = [e for e in NET if e[0] == 'RL']; nb = len(branches); nn = len(node_names); n = nn + nb
Em = np.zeros((n, n)); Gm = np.zeros((n, n)); Bm = np.zeros((n, 2))
def stamp(M, a, b, g):                       # two-terminal stamp on node rows/cols (ground dropped)
    for p, q, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
        if p != '0' and q != '0': M[nidx[p], nidx[q]] += s*g
for e in NET:
    if e[0] == 'R': stamp(Gm, e[2], e[3], 1.0/e[4])
    elif e[0] == 'C': stamp(Em, e[2], e[3], e[4])
    elif e[0] == 'I':                         # current from node+ through the source to node-: injects +I into node-
        if e[3] != '0': Bm[nidx[e[3]], e[4]] += 1.0
        if e[2] != '0': Bm[nidx[e[2]], e[4]] -= 1.0
for k, e in enumerate(branches):              # branch k: current i_k flows from e[2] to e[3]
    row = nn + k
    if e[2] != '0': Gm[nidx[e[2]], row] += 1.0; Gm[row, nidx[e[2]]] -= 1.0
    if e[3] != '0': Gm[nidx[e[3]], row] -= 1.0; Gm[row, nidx[e[3]]] += 1.0
    Em[row, row] = e[4]; Gm[row, row] += e[5]
# MNA system: Em x' + Gm x = Bm u   ->  descriptor form E x' = A x + B u with A = -Gm
Am = -Gm
# map MNA ordering [v1 v2 v3 va vb i1 i2] to hand ordering [v1 v2 v3 i1 i2 va vb]
perm = [0, 1, 2, 5, 6, 3, 4]                  # hand index h <- MNA index perm[h]
Ep, Ap, Bp = Em[np.ix_(perm, perm)], Am[np.ix_(perm, perm)], Bm[perm]
# hand-derived matrices have row order [KCL1 KCL2 KCL3 KVL1 KVL2 alg_a alg_b] = same permutation of MNA rows
dE, dA, dB = (np.abs(Ep - nc.E).max()/np.abs(nc.E).max(), np.abs(Ap - nc.A).max()/np.abs(nc.A).max(), np.abs(Bp - nc.B).max())
# ---------------- independent BVP solution: Chebyshev collocation on the MNA system (own implementation) ----------------
def chebD(N):
    j = np.arange(N + 1); x = np.cos(np.pi*j/N); c = np.where((j == 0) | (j == N), 2.0, 1.0)*(-1.0)**j
    D = np.zeros((N + 1, N + 1))
    for p in range(N + 1):
        for q in range(N + 1):
            if p != q: D[p, q] = c[p]/(c[q]*(x[p] - x[q]))
    D -= np.diag(D.sum(1)); return D, x
def src(t, j):
    arg = 2*np.pi*t/nc.T + nc.PH[j]; return np.stack([nc.AMP[j]*(0.8 + 0.3*np.sin(arg)), 0.1 + 0.05*np.cos(arg)])      # (2, len(t))
N = 70; D, x = chebD(N); t = nc.T*(1 - x)/2; Dt = -(2/nc.T)*D                    # j=0 <-> t=0, j=N <-> t=T
Xm = []
for j in range(3):
    K = np.kron(Em, Dt) - np.kron(Am, np.eye(N + 1)); rhs = (Bm @ src(t, j)).reshape(-1)       # unknown ordering: component-major (n blocks of N+1)
    for comp in range(3): r = comp*(N + 1) + 0; K[r] = 0; K[r, r] = 1; rhs[r] = nc.V0[comp, j]                           # v1,v2,v3 at t=0
    for k in range(2): r = (nn + k)*(N + 1) + N; K[r] = 0; K[r, r] = 1; rhs[r] = nc.IT[k, j]                             # i1,i2 at t=T
    Xm.append(np.linalg.solve(K, rhs).reshape(n, N + 1))
# ---------------- comparison on a common grid ----------------
tq = np.linspace(0, nc.T, 401); ref = nc.reference(tq)            # hand-derived reference, hand order, (7,401,3)
from scipy.interpolate import BarycentricInterpolator
def evalm(j, comp, tt, deriv=False):
    f = BarycentricInterpolator(t[::-1], Xm[j][comp][::-1]); return f.derivative(tt) if deriv else f(tt)
out = dict(KR=kR, KL=kL, KC=kC, rel_diff_E=float(dE), rel_diff_A=float(dA), abs_diff_B=float(dB))
mn = [np.stack([evalm(j, perm[h], tq) for h in range(7)]) for j in range(3)]     # MNA solution in hand order
mn = np.stack(mn, -1)                                                            # (7,401,3)
out['max_abs_MNA_vs_hand_ref'] = float(np.abs(mn - ref).max()); out['rel_L2_MNA_vs_hand_ref'] = float(np.linalg.norm(mn - ref)/np.linalg.norm(ref))
hc = []
for j in range(3):
    th, Xh, _ = nc.cheb_full_descriptor(j, 70); hc.append(np.stack([BarycentricInterpolator(th[::-1], Xh[k][::-1])(tq) for k in range(7)]))
hc = np.stack(hc, -1)
out['max_abs_MNA_vs_handCheb'] = float(np.abs(mn - hc).max()); out['max_abs_handCheb_vs_handRef'] = float(np.abs(hc - ref).max())
# residuals of the MNA equations (permuted back to MNA ordering) evaluated on the MNA-Chebyshev solution (spectral derivative) and on the hand reference
def residuals(sol_fun, dsol_fun):
    res = {}
    kcl, kvl, alg = [], [], []
    for j in range(3):
        xx = np.stack([sol_fun(j, c) for c in range(n)]); dx = np.stack([dsol_fun(j, c) for c in range(n)])      # MNA ordering (n, 401)
        r = Em @ dx - Am @ xx - Bm @ src(tq, j)
        kcl.append(np.abs(r[:3]).max()); alg.append(np.abs(r[3:5]).max()); kvl.append(np.abs(r[5:]).max())
    return dict(KCL=float(max(kcl)), KVL=float(max(kvl)), algebraic=float(max(alg)))
out['residual_on_MNA_solution'] = residuals(lambda j, c: evalm(j, c, tq), lambda j, c: evalm(j, c, tq, True))
inv = np.argsort(perm)                                                              # MNA index c -> hand index h with perm[h]=c
def hand_sol(j, c): return ref[inv[c], :, j]
def hand_dsol(j, c):
    xd = nc._SOL[j].sol(tq); dxd = np.stack([nc._f(tt, xd[:, i], j) for i, tt in enumerate(tq)], 1); dxa = nc.M_REC @ dxd
    return np.vstack([dxd, dxa])[inv[c]]
out['residual_on_hand_reference'] = residuals(hand_sol, hand_dsol)
out['BC_residual_MNA'] = float(max(max(abs(evalm(j, c, 0.0) - nc.V0[c, j]) for c in range(3)) for j in range(3)) + 0)
out['BC_residual_MNA_i'] = float(max(max(abs(evalm(j, nn + k, nc.T) - nc.IT[k, j]) for k in range(2)) for j in range(3)))
print(json.dumps(out))
