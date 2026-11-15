"""Validated sup-norm residual enclosure for trained PINNs (hard BC) and the resulting error bound for the coupled RLC descriptor BVP.
Interval arithmetic in centre-radius form with a conservative rounding margin after every operation (not a formally verified library).
Mean-value form on K subintervals: |r(t)| <= |r(c)| + sup|r_tau(I)| * w.  Usage: SENS=kR,kL,kC python3 cert.py out.jsonl seed0 seed1"""
import sys, json, numpy as np, scipy.linalg as sla
import network_pinn as npn, network_core as nc
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
EPS = 4e-16
def fix(m, r): return m, r + EPS*(np.abs(m) + r) + 1e-300
def imul(a, b):
    m = a[0]*b[0]; r = np.abs(a[0])*b[1] + np.abs(b[0])*a[1] + a[1]*b[1]; return fix(m, r)
def iadd(a, b): return fix(a[0] + b[0], a[1] + b[1])
def isub(a, b): return fix(a[0] - b[0], a[1] + b[1])
def iscale(a, c): return fix(a[0]*c, a[1]*np.abs(c))
def itanh(a):
    lo, hi = np.tanh(a[0] - a[1]), np.tanh(a[0] + a[1]); m = 0.5*(lo + hi); r = 0.5*(hi - lo) + 2e-16; return fix(m, r + 1e-15)
def imatvec(W, a, axis_last=True):                     # a: (..., q) interval, W: (p, q) point -> (..., p)
    m = a[0] @ W.T; absW = np.abs(W); r = a[1] @ absW.T + 1e-13*(np.abs(a[0]) @ absW.T) + 1e-300; return fix(m, r)
def imat_cols(M, a):                                   # M: (p,q) point, a: (K,q,3) interval -> (K,p,3)
    m = np.einsum('pq,kqc->kpc', M, a[0]); r = np.einsum('pq,kqc->kpc', np.abs(M), a[1]) + 1e-13*np.einsum('pq,kqc->kpc', np.abs(M), np.abs(a[0])) + 1e-300; return fix(m, r)
def forward(ps, c, w, nout):
    """network and its tau-derivatives (1st, 2nd) on intervals tau in [c-w, c+w]; c, w: (K,)"""
    K = len(c); h = (c[:, None], w[:, None]); h1 = (np.ones((K, 1)), np.zeros((K, 1))); h2 = (np.zeros((K, 1)), np.zeros((K, 1)))
    for W, b in ps[:-1]:
        z = imatvec(W, h); z = (z[0] + b, z[1])
        z1 = imatvec(W, h1); z2 = imatvec(W, h2)
        a = itanh(z); s = isub((np.ones_like(a[0]), np.zeros_like(a[1])), imul(a, a))
        a1 = imul(s, z1)
        a2 = isub(imul(s, z2), iscale(imul(imul(a, s), imul(z1, z1)), 2.0))
        h, h1, h2 = a, a1, a2
    W, b = ps[-1]
    out = [imatvec(W, q) for q in (h, h1, h2)]; out[0] = (out[0][0] + b, out[0][1])
    return [(o[0].reshape(K, nout, 3), o[1].reshape(K, nout, 3)) for o in out]
def assemble(ps, c, w, variant):
    nout = 7 if variant == 'D' else 5
    N, N1, N2 = forward(ps, c, w, nout)
    P0 = forward(ps, np.array([0.0]), np.array([0.0]), nout)[0]; P1 = forward(ps, np.array([1.0]), np.array([0.0]), nout)[0]
    K = len(c); tau = (c[:, None, None], w[:, None, None])
    d0 = (np.asarray(npn.V0s)[None] - P0[0][:, :3], P0[1][:, :3]); d1 = (np.asarray(npn.ITs)[None] - P1[0][:, 3:5], P1[1][:, 3:5])
    omt = (1 - c[:, None, None], w[:, None, None])
    sl = lambda q, a, b: (q[0][:, a:b], q[1][:, a:b])
    v = iadd(sl(N, 0, 3), imul(omt, (np.broadcast_to(d0[0], (K, 3, 3)), np.broadcast_to(d0[1], (K, 3, 3)))))
    v1 = isub(sl(N1, 0, 3), (np.broadcast_to(d0[0], (K, 3, 3)), np.broadcast_to(d0[1], (K, 3, 3))))
    i = iadd(sl(N, 3, 5), imul(tau, (np.broadcast_to(d1[0], (K, 2, 3)), np.broadcast_to(d1[1], (K, 2, 3)))))
    i1 = iadd(sl(N1, 3, 5), (np.broadcast_to(d1[0], (K, 2, 3)), np.broadcast_to(d1[1], (K, 2, 3))))
    cat = lambda a, b: (np.concatenate([a[0], b[0]], 1), np.concatenate([a[1], b[1]], 1))
    xd, xd1, xd2 = cat(v, i), cat(v1, i1), sl(N2, 0, 5)
    xd2 = (xd2[0], xd2[1])
    if variant == 'D': return cat(xd, sl(N, 5, 7)), cat(xd1, sl(N1, 5, 7)), cat(xd2, sl(N2, 5, 7))
    S5 = np.asarray(npn.S[:5]); Mr = np.asarray(npn.MREC)/np.asarray(npn.S[5:])[:, None]*1.0
    Mr = np.asarray(nc.M_REC)*S5[None, :]/np.asarray(npn.S[5:])[:, None]
    tr = lambda q: imat_cols(Mr, q)
    return cat(xd, tr(xd)), cat(xd1, tr(xd1)), cat(xd2, tr(xd2))
def residual_enclosure(ps, variant, K=2000):
    edges = np.linspace(0, 1, K + 1); c = 0.5*(edges[1:] + edges[:-1]); w = 0.5*np.diff(edges)
    x, x1, x2 = assemble(ps, c, np.zeros_like(c), variant)                 # point values at centres
    _, y1, y2 = assemble(ps, c, w, variant)                                  # derivatives on intervals
    Sv = np.asarray(npn.S)[None, :, None]; T = nc.T; E, A, B = nc.E, nc.A, nc.B
    def u(tc):
        PH, AMP = nc.PH, nc.AMP; arg = 2*np.pi*tc[:, None] + PH[None]
        u1 = AMP*(0.8 + 0.3*np.sin(arg)); u3 = 0.1 + 0.05*np.cos(arg); return np.stack([u1, u3], 1)       # (K,2,3)
    def du(tc):
        PH, AMP = nc.PH, nc.AMP; arg = 2*np.pi*tc[:, None] + PH[None]
        return np.stack([AMP*0.3*2*np.pi*np.cos(arg), -0.05*2*np.pi*np.sin(arg)], 1)
    # r(c) = E (S x1)/T - A (S x) - B u(c);  point values evaluated with interval margins
    def lin(M, q): return imat_cols(M, q)
    Sx, Sx1 = iscale(x, Sv), iscale(x1, Sv/T)
    # x here is the first output of assemble at zero width; x1 (second) holds tau-derivative at centres
    x0, d1c, _ = assemble(ps, c, np.zeros_like(c), variant)
    r_c = isub(lin(E, iscale(d1c, Sv/T)), lin(A, iscale(x0, Sv)))
    uc = u(c); r_c = isub(r_c, (np.einsum('pq,kqc->kpc', B, uc), 1e-13*np.einsum('pq,kqc->kpc', np.abs(B), np.abs(uc)) + 1e-300))
    # r_tau on intervals
    _, d1I, d2I = assemble(ps, c, w, variant)
    xI, _, _ = assemble(ps, c, w, variant)
    r_t = isub(lin(E, iscale(d2I, Sv/T)), lin(A, iscale(d1I, Sv)))
    ud = du(c); ud_r = (2*np.pi)**2*np.array([0.3*nc.AMP.max(), 0.05])[None, :, None]*w[:, None, None]   # |u''| <= (2 pi)^2 * amplitude
    r_t = isub(r_t, (np.einsum('pq,kqc->kpc', B, ud), np.einsum('pq,kqc->kpc', np.abs(B), np.broadcast_to(ud_r, ud.shape)) + 1e-13*np.einsum('pq,kqc->kpc', np.abs(B), np.abs(ud)) + 1e-300))
    return r_c, r_t, w
def constants(Ns=1500, Nt=300):
    """K_d = sup_t [ int_0^T ||Phi(t) Jz Phi_ii(T)^-1 Phi_i.(T-s)||_2 ds + int_0^t ||Phi(t-s)||_2 ds ] bounded from above.
    Exponentials in 40-digit arithmetic (mpmath), then cell bounds: ||expm(M d) X|| <= exp(||M||_2 |d|) ||X|| on each cell, so a midpoint sum times
    exp(||M||_2 h/2) is an upper bound of the integral; the sup over t is taken over Nt cells in the same way; the second integral is nondecreasing in t
    so its supremum is the value at T. Spectral norms by LAPACK SVD, inflated by 1e-9 (not interval-verified)."""
    import mpmath as mp
    mp.mp.dps = 40
    E = nc.E; Ed = E[:5, :5]; A12, A21, A22 = nc.A12, nc.A21, nc.A22
    Ai = np.linalg.inv(A22); G = np.hstack([np.linalg.inv(Ed), -np.linalg.inv(Ed) @ A12 @ Ai])
    T = nc.T; Mred = nc.Mred; mn = np.linalg.norm(Mred, 2)
    Mm = mp.matrix(Mred.tolist())
    ex = lambda u: np.array(mp.expm(Mm*mp.mpf(float(u))).tolist(), dtype=float)
    hs = T/Ns; sc = (np.arange(Ns) + 0.5)*hs                     # midpoints of s-cells (and u-cells)
    PhiTs = np.array([ex(T - x) for x in sc])                       # Phi(T - s_c)
    PhiU = np.array([ex(x) for x in sc])                            # Phi(u_c)
    PT = ex(T); Pinv = np.array(mp.matrix(PT[3:5, 3:5].tolist())**-1, dtype=float) if False else np.linalg.inv(PT[3:5, 3:5])
    # inflate: the inverse and all exponentials are rounded to double (relative error ~1e-16); covered by the 1e-9 inflation of the norms
    infl = 1 + 1e-9
    I2 = hs*np.exp(mn*hs/2)*infl*np.linalg.norm(PhiU, 2, axis=(1, 2)).sum()
    Jz = np.zeros((5, 2)); Jz[3:5] = np.eye(2)
    H = np.einsum('pq,sqr->spr', Pinv, PhiTs[:, 3:5, :])           # (Ns,2,5) = Phi_ii(T)^-1 Phi_i.(T-s_c)
    ht = T/Nt; tc = (np.arange(Nt) + 0.5)*ht; best = 0.0
    for ta in tc:
        Pt = ex(ta)[:, 3:5]                                            # Phi(t_a) Jz  (5x2)
        P = np.einsum('pq,sqr->spr', Pt, H)                            # (Ns,5,5)
        I1 = hs*np.exp(mn*hs/2)*np.linalg.norm(P, 2, axis=(1, 2)).sum()*infl*np.exp(mn*ht/2)
        best = max(best, I1)
    return dict(G=G, Kd=best + I2, Kd_parts=(best, I2), Ainv=np.linalg.norm(Ai, 2), A21=np.linalg.norm(A21, 2), Ed=Ed)
def constants_deriv(Ns=3000, Nt=600):
    """Second, independent route to an upper bound of K_d that avoids the exponential cell inequality: Lipschitz (derivative) bounds.
    F(s)=S_i Phi(T-s) has F'(s) = -S_i M Phi(T-s), so ||F(s')-F(s_c)|| <= ||M||_2 Phimax |s'-s_c|; likewise ||Phi(t')-Phi(t_a)|| <= ||M||_2 Phimax |t'-t_a|.
    P(t',s') - P(t_a,s_c) = [Phi(t')-Phi(t_a)] J Pinv F(s') + Phi(t_a) J Pinv [F(s')-F(s_c)], hence
    ||P(t',s')|| <= ||P(t_a,s_c)|| + ||Pinv|| Phimax^2 ||M||_2 (h_t+h_s)/2 on every cell. Phimax bounds ||Phi(u)||_2 on [0,T] from the midpoint maximum times exp(||M||_2 h_s/2) (semigroup inequality)."""
    import mpmath as mp
    mp.mp.dps = 40
    T = nc.T; Mred = nc.Mred; mn = np.linalg.norm(Mred, 2); Mm = mp.matrix(Mred.tolist())
    ex = lambda u: np.array(mp.expm(Mm*mp.mpf(float(u))).tolist(), dtype=float)
    hs = T/Ns; sc = (np.arange(Ns) + 0.5)*hs
    PhiTs = np.array([ex(T - x) for x in sc]); PhiU = np.array([ex(x) for x in sc])
    infl = 1 + 1e-9
    nU = np.linalg.norm(PhiU, 2, axis=(1, 2)); Phimax = nU.max()*np.exp(mn*hs/2)*infl       # sup ||Phi(u)||_2 <= max_j ||Phi(u_j)||_2 exp(||M||_2 hs/2) by the semigroup inequality (no first-order truncation)
    PT = ex(T); Pinv = np.linalg.inv(PT[3:5, 3:5]); pn = np.linalg.norm(Pinv, 2)*infl
    I2 = hs*(nU + mn*Phimax*hs/2).sum()*infl
    H = np.einsum('pq,sqr->spr', Pinv, PhiTs[:, 3:5, :])
    ht = T/Nt; tc = (np.arange(Nt) + 0.5)*ht; best = 0.0; dP = pn*Phimax**2*mn*(ht + hs)/2
    for ta in tc:
        P = np.einsum('pq,sqr->spr', ex(ta)[:, 3:5], H)
        best = max(best, hs*(np.linalg.norm(P, 2, axis=(1, 2))*infl + dP).sum())
    return dict(Kd=best + I2, Kd_parts=(best, I2), Phimax=Phimax, Pinv=pn)
CONST = None
def certify(ps, variant, K=2000):
    global CONST
    if CONST is None:
        CONST = constants(); d = constants_deriv(); CONST['Kd_exp'] = CONST['Kd']; CONST['Kd_deriv'] = d['Kd']; CONST['Kd'] = max(CONST['Kd'], d['Kd'])   # use the larger of the two bounds
    r_c, r_t, w = residual_enclosure(ps, variant, K)
    G = CONST['G']
    def up(r):                                                      # upper bound of |component| on each subinterval, shape (K, 7, 3)
        return np.abs(r_c[0]) + r_c[1]
    ub = np.abs(r_c[0]) + r_c[1] + (np.abs(r_t[0]) + r_t[1])*w[:, None, None]       # |r_i(t)| bound per interval and component
    # g = G r: bound via the same mean-value form applied to the linear image
    g_c = (np.einsum('gp,kpc->kgc', G, r_c[0]), np.einsum('gp,kpc->kgc', np.abs(G), r_c[1]))
    g_t = (np.einsum('gp,kpc->kgc', G, r_t[0]), np.einsum('gp,kpc->kgc', np.abs(G), r_t[1]))
    gub = np.abs(g_c[0]) + g_c[1] + (np.abs(g_t[0]) + g_t[1])*w[:, None, None]
    gnorm = np.sqrt((gub**2).sum(1)).max(0)                         # per column sup ||g||_2 upper bound (K x 5 x 3 -> 3)
    ra = np.sqrt((ub[:, 5:]**2).sum(1)).max(0)                      # per column sup ||r_a||_2
    rd = np.sqrt((ub[:, :5]**2).sum(1)).max(0)
    ed = CONST['Kd']*gnorm; ea = CONST['Ainv']*(CONST['A21']*ed + ra)
    return dict(bound=np.sqrt(ed**2 + ea**2), sup_g=gnorm, sup_ra=ra, sup_rd=rd)
def train(variant, seed, N=2000):
    unr, lg, step, evaluate, nout = npn.build(variant, True)
    flat = ravel_pytree(npn.init(__import__('jax').random.PRNGKey(seed), nout=nout))[0]
    fun = lambda x: tuple(np.asarray(a, dtype=np.float64) for a in lg(jnp.asarray(x)))
    r = minimize(fun, np.asarray(flat), jac=True, method='L-BFGS-B', options=dict(maxfun=N, maxiter=N, maxcor=50, ftol=1e-18, gtol=1e-14))
    return unr, jnp.asarray(r.x), evaluate
if __name__ == "__main__":
    fn = sys.argv[1]; seeds = [int(a) for a in sys.argv[2:]]
    for seed in seeds:
        for variant in 'RD':
            unr, flat, evaluate = train(variant, seed)
            ps = [(np.asarray(W), np.asarray(b)) for W, b in unr(flat)]
            tq = np.linspace(0, 1, 5001); refd = nc.reference(tq*nc.T)
            import jax
            xs = np.asarray(jax.vmap(lambda t: npn.state(unr(flat), t, variant, True))(jnp.asarray(tq))); X = np.transpose(xs, (1, 0, 2))*npn.S[:, None, None]; e = X - refd   # sampled on 5001 points
            true_sup = np.sqrt((e**2).sum(0)).max(0)                          # sampled sup_t ||e||_2 (5001 points, not an exact supremum)
            c = certify(ps, variant)
            # loss-based quantity: rms of the physical residual on the 101 training points (all comps, per column) 
            import jax
            rr = np.asarray(jax.vmap(lambda t: npn.residual(unr(flat), t, variant, True))(jnp.asarray(npn.TAU_C)))   # (101,7,3)
            rms = np.sqrt((rr**2).sum(1).mean(0))
            rec = dict(variant=variant, seed=seed, KR=nc.KR, KL=nc.KL, KC=nc.KC, true_sup_err_sampled5001=true_sup.tolist(), bound=c['bound'].tolist(), sup_g=c['sup_g'].tolist(),
                       sup_ra=c['sup_ra'].tolist(), sup_rd=c['sup_rd'].tolist(), train_rms_residual=rms.tolist(), ratio=(c['bound']/true_sup).tolist(), Kd=CONST['Kd'], Kd_exp=CONST['Kd_exp'], Kd_deriv=CONST['Kd_deriv'])
            open(fn, 'a').write(json.dumps(rec) + '\n')
            print(variant, seed, 'true', np.round(true_sup, 5), 'bound', np.round(c['bound'], 4), 'ratio', np.round(c['bound']/true_sup, 1), flush=True)
