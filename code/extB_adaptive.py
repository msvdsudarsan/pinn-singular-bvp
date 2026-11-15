"""Extension B: fixed versus residual-adaptive collocation under an matched nominal evaluation budget.
Budget N=2000 loss+gradient evaluations in ROUNDS rounds of L-BFGS-B (every method uses the same restarts and 101 collocation points).
methods: fixed (uniform points every round) | adapt (RAD-type resampling, indicator = scaled residual max over rows)
         | adaptw (indicator = G1*|r_diff| + G2*|r_alg|, the bound-weighted indicator; identical to adapt when r_alg = 0)
Usage: [RAB=..] [KT=..] python3 extB_adaptive.py <seed0> <seed1> <out.jsonl>"""
import sys, json, time, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
import network_core as nc, network_pinn as npn
import extA_separation as ea
S, RS = npn.S, npn.RS; ROUNDS = 4; NB = 2000; NC = 101; CAND = np.linspace(0, 1, 1001)
G1, G2 = ea.green_norms()
CACHEB = {}
def build(variant):
    if variant in CACHEB: return CACHEB[variant]
    nout = 7 if variant == 'D' else 5
    flat0, unr = ravel_pytree(npn.init(jax.random.PRNGKey(0), nout=nout))
    def loss(f, tc):
        ps = unr(f); r = jax.vmap(lambda t: npn.residual(ps, t, variant, True))(tc)*npn.RSj[None, :, None]
        return jnp.mean(r[:, :5]**2) + jnp.mean(r[:, 5:]**2)
    lg = jax.jit(jax.value_and_grad(loss))
    @jax.jit
    def resid(f, tq):
        ps = unr(f); return jax.vmap(lambda t: npn.residual(ps, t, variant, True))(tq)*npn.RSj[None, :, None]
    @jax.jit
    def states(f, tq):
        ps = unr(f); return jax.vmap(lambda t: npn.state(ps, t, variant, True))(tq)
    CACHEB[variant] = (unr, lg, resid, states, nout); return CACHEB[variant]
def indicator(rr, method):
    ad = np.abs(rr[:, :5]).max(axis=(1, 2)); aa = np.abs(rr[:, 5:]).max(axis=(1, 2))
    return np.maximum(ad, aa) if method == 'adapt' else G1*ad + G2*aa
def run(variant, method, seed):
    unr, lg, resid, states, nout = build(variant)
    flat = np.asarray(ravel_pytree(npn.init(jax.random.PRNGKey(seed), nout=nout))[0]); rng = np.random.default_rng(1000 + seed)
    tc = np.linspace(0, 1, NC); used = 0; t0 = time.time(); hist = []
    for k in range(ROUNDS):
        fun = lambda x, tc=tc: tuple(np.asarray(a, dtype=np.float64) for a in lg(jnp.asarray(x), jnp.asarray(tc)))
        r = minimize(fun, flat, jac=True, method='L-BFGS-B', options=dict(maxfun=NB//ROUNDS, maxiter=NB//ROUNDS, maxcor=50, ftol=1e-18, gtol=1e-14))
        flat = r.x; used += int(r.nfev)
        if k < ROUNDS-1 and method != 'fixed':
            eta = indicator(np.asarray(resid(jnp.asarray(flat), jnp.asarray(CAND))), method)
            p = eta/eta.mean() + 1.0; p /= p.sum()
            idx = rng.choice(len(CAND), size=NC-2, replace=False, p=p)
            tc = np.sort(np.r_[0.0, 1.0, CAND[idx]])
    tq = np.linspace(0, 1, 501)
    xs = np.asarray(states(jnp.asarray(flat), jnp.asarray(tq))); X = np.transpose(xs, (1, 0, 2))*S[:, None, None]; e = X - npn.REF
    rr = np.asarray(resid(jnp.asarray(flat), jnp.asarray(tq)))
    ad = float(np.abs(rr[:, :5]).max()); aa = float(np.abs(rr[:, 5:]).max()); es = e/S[:, None, None]
    out = dict(variant=variant, method=method, seed=seed, rab=nc.RAB, T=nc.T, relL2=float(np.linalg.norm(e)/np.linalg.norm(npn.REF)),
               sup_err=float(np.abs(es).max()), sup_r1=ad, sup_r2=aa, nfev=used, sec=time.time()-t0)
    out['fail'] = bool(not np.isfinite(out['relL2']) or out['relL2'] > 5e-2); return out
if __name__ == "__main__":
    s0, s1, fn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    for seed in range(s0, s1):
        for variant, methods in (('D', ('fixed', 'adapt', 'adaptw')), ('R', ('fixed', 'adapt'))):
            for m in methods:
                o = run(variant, m, seed); open(fn, 'a').write(json.dumps(o) + '\n')
                print(variant, m, seed, f"{o['relL2']:.2e}", 'FAIL' if o['fail'] else '', f"{o['sec']:.0f}s", flush=True)
