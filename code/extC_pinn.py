"""Extension C: full (D) versus reduced (R) PINN on the n-node RLC ladder (hard boundary conditions, matched nominal evaluation budget).
Usage: NNODE=<n> python3 extC_pinn.py <seed0> <seed1> <N> <out.jsonl>"""
import os, sys, json, time, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
import ladder_core as lc
n, nd, N, na, T = lc.n, lc.nd, lc.N, lc.na, lc.T
VS, IS = lc.VSC, lc.ISC
S = np.array([VS]*n + [IS]*(n-1) + [VS]*na); RS = np.array([1/IS]*n + [1/VS]*(n-1) + [1/IS]*na)
Ej, Aj, Bj, Sj, RSj, MREC = (jnp.array(a) for a in (lc.E, lc.A, lc.B, S, RS, lc.M_REC))
V0s, ITs = jnp.array(lc.V0/VS), jnp.array(lc.IT/IS)
TAU_C = jnp.linspace(0.0, 1.0, 101); tau_eval = np.linspace(0, 1, 501); REF = lc.reference(tau_eval*T)
def init(key, w, d, nout):
    sizes = [1] + [w]*d + [nout]; ps = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key); ps.append((jax.random.normal(k, (b, a))*np.sqrt(2/(a+b)), jnp.zeros(b)))
    return ps
def raw(ps, tau):
    h = jnp.atleast_1d(tau)
    for W, b in ps[:-1]: h = jnp.tanh(W@h + b)
    W, b = ps[-1]; return W@h + b
def state(ps, tau, variant):
    Nn = raw(ps, tau); N0, N1 = raw(ps, 0.0), raw(ps, 1.0)
    v = Nn[:n] + (1 - tau)*(V0s - N0[:n]); i = Nn[n:nd] + tau*(ITs - N1[n:nd]); xd = jnp.concatenate([v, i])
    if variant == 'D': return jnp.concatenate([xd, Nn[nd:]])
    return jnp.concatenate([xd, (MREC @ (Sj[:nd]*xd))/Sj[nd:]])
def sources(tau): return jnp.array([0.8 + 0.3*jnp.sin(2*jnp.pi*tau), 0.1 + 0.05*jnp.cos(2*jnp.pi*tau)])
def residual(ps, tau, variant):
    xs, dxs = jax.jvp(lambda s: state(ps, s, variant), (tau,), (1.0,))
    return (Ej @ (Sj*dxs/T) - Aj @ (Sj*xs) - Bj @ sources(tau))*RSj
def run(variant, mode, seed, Nb, w=64, d=4):
    nout = nd if variant == 'R' else N
    flat, unr = ravel_pytree(init(jax.random.PRNGKey(seed), w, d, nout))
    def loss(f):
        ps = unr(f); r = jax.vmap(lambda t: residual(ps, t, variant))(TAU_C)
        return jnp.mean(r[:, :nd]**2) + jnp.mean(r[:, nd:]**2)
    lg = jax.jit(jax.value_and_grad(loss))
    @jax.jit
    def step(f, m, v, k):
        l, g = jax.value_and_grad(loss)(f); m = .9*m + .1*g; v = .999*v + .001*g**2
        return f - 1e-3*(m/(1 - .9**k))/(jnp.sqrt(v/(1 - .999**k)) + 1e-8), m, v, l
    na_ = 0 if mode == 'lbfgs' else int(0.3*Nb); t0 = time.time(); m = jnp.zeros_like(flat); v = jnp.zeros_like(flat)
    for k in range(1, na_+1): flat, m, v, l = step(flat, m, v, float(k))
    fun = lambda x: tuple(np.asarray(a, dtype=np.float64) for a in lg(jnp.asarray(x)))
    r = minimize(fun, np.asarray(flat), jac=True, method='L-BFGS-B', options=dict(maxfun=Nb-na_, maxiter=Nb-na_, maxcor=50, ftol=1e-18, gtol=1e-14))
    ps = unr(jnp.asarray(r.x)); xs = np.asarray(jax.vmap(lambda t: state(ps, t, variant))(jnp.asarray(tau_eval)))
    X = xs.T*S[:, None]; e = X - REF
    rr = np.asarray(jax.vmap(lambda t: residual(ps, t, variant))(jnp.asarray(tau_eval)))
    o = dict(nfev=int(r.nfev), status=int(r.status), message=str(r.message), n=n, states=N, variant=variant, mode=mode, seed=seed, budget=Nb, relL2=float(np.linalg.norm(e)/np.linalg.norm(REF)),
             relL2_alg=float(np.linalg.norm(e[nd:])/np.linalg.norm(REF[nd:])), sup_r_diff=float(np.abs(rr[:, :nd]).max()),
             sup_r_alg=float(np.abs(rr[:, nd:]).max()), sec=time.time()-t0, condA22=float(np.linalg.cond(lc.A22)), rab=lc.RAB)
    o['fail'] = bool(not np.isfinite(o['relL2']) or o['relL2'] > 5e-2); return o
if __name__ == "__main__":
    s0, s1, Nb, fn = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    for seed in range(s0, s1):
        for variant in ('D', 'R'):
            for mode in os.environ.get('MODES','lbfgs,two').split(','):
                o = run(variant, mode, seed, Nb); open(fn, 'a').write(json.dumps(o) + '\n')
                print(n, variant, mode, seed, f"{o['relL2']:.2e}", 'FAIL' if o['fail'] else '', f"{o['sec']:.0f}s", flush=True)
