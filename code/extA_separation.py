"""Extension A: conditioning-dependent separation of Variants D and R.
Part 1: discrete Green's operator norms ||G1||, ||G2|| (differential and algebraic residual rows -> error) and ||A22^{-1}||.
Part 2 (trained networks): check  sup-error <= ||G1|| sup|r1| + ||G2|| sup|r2|  on sampled grids and record how tight it is.
Usage: RAB=<scale> python3 extA_separation.py <seed0> <seed1> <out.jsonl>"""
import sys, json, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import network_core as nc, network_pinn as npn

S, RS, T = npn.S, npn.RS, nc.T

def green_norms(N=80):
    """Induced sup-norm (max abs row sum, scaled units) of the discrete solution map r -> e for E e' = A e + r, hard BC homogeneous."""
    D, x = nc.cheb(N); Dt = -(2/T)*D
    M = np.kron(nc.E, Dt) - np.kron(nc.A, np.eye(N+1))
    bc = [r*(N+1) for r in range(3)] + [r*(N+1)+N for r in (3, 4)]
    n = 7*(N+1); Minv_cols = np.zeros((n, n))
    Mb = M.copy()
    for i in bc: Mb[i] = 0; Mb[i, i] = 1
    Mi = np.linalg.inv(Mb)
    Rop = np.eye(n)
    for i in bc: Rop[i, i] = 0                    # boundary rows carry no residual
    G = Mi @ Rop                                  # e = G r (physical units, stacked state-major)
    # scaled: e_s = e/S, r_s = r*RS  ->  e_s = diag(1/S) G diag(1/RS) r_s
    sc_e = np.repeat(1/S, N+1); sc_r = np.repeat(1/RS, N+1)
    Gs = (sc_e[:, None]*G)*sc_r[None, :]
    rows = np.arange(n)
    cols_d = np.concatenate([np.arange(k*(N+1), (k+1)*(N+1)) for k in range(5)])
    cols_a = np.concatenate([np.arange(k*(N+1), (k+1)*(N+1)) for k in (5, 6)])
    g1 = np.abs(Gs[:, cols_d]).sum(1).max(); g2 = np.abs(Gs[:, cols_a]).sum(1).max()
    return float(g1), float(g2)
def a22_inv_scaled():
    # pointwise amplification of an algebraic residual into the algebraic states, scaled units
    Ai = np.linalg.inv(nc.A22); Ws = Ai*(1/S[5:])[:, None]*(1/RS[5:])[None, :]
    return float(np.abs(Ws).sum(1).max())

def run2(variant, seed, N=2000):
    unr, lg, step, evaluate, nout = npn.build(variant, True)
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    flat = ravel_pytree(npn.init(jax.random.PRNGKey(seed), nout=nout))[0]
    fun = lambda x: tuple(np.asarray(a, dtype=np.float64) for a in lg(jnp.asarray(x)))
    r = minimize(fun, np.asarray(flat), jac=True, method='L-BFGS-B', options=dict(maxfun=N, maxiter=N, maxcor=50, ftol=1e-18, gtol=1e-14))
    xs, res, x0, x1, pr = evaluate(jnp.asarray(r.x)); xs = np.asarray(xs); res = np.asarray(res)
    X = np.transpose(xs, (1, 0, 2))*S[:, None, None]; e = X - npn.REF
    es = e/S[:, None, None]; rr = np.transpose(res, (1, 0, 2))*RS[:, None, None]
    return dict(variant=variant, seed=seed, rab=nc.RAB, relL2=float(np.linalg.norm(e)/np.linalg.norm(npn.REF)),
        sup_err=float(np.abs(es).max()), sup_r1=float(np.abs(rr[:5]).max()), sup_r2=float(np.abs(rr[5:]).max()),
        sup_err_alg=float(np.abs(es[5:]).max()), nfev=int(r.nfev))
if __name__ == "__main__":
    s0, s1, fn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    g1, g2 = green_norms(); ainv = a22_inv_scaled()
    print('RAB', nc.RAB, 'G1', g1, 'G2', g2, 'A22inv_scaled', ainv, 'condA22', np.linalg.cond(nc.A22), flush=True)
    for seed in range(s0, s1):
        for v in ('D', 'R'):
            o = run2(v, seed); o.update(G1=g1, G2=g2, A22inv=ainv, condA22=float(np.linalg.cond(nc.A22)))
            o['bound'] = g1*o['sup_r1'] + g2*o['sup_r2']; o['holds'] = bool(o['sup_err'] <= o['bound']*(1+1e-9))
            open(fn, 'a').write(json.dumps(o) + '\n'); print(v, seed, f"{o['relL2']:.2e} err {o['sup_err']:.2e} bound {o['bound']:.2e} holds {o['holds']}", flush=True)
