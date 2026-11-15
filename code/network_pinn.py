"""Experiment B: PINN for the three-node coupled RLC descriptor BVP.
Variant D : network outputs all 7 states, separate differential and algebraic residual losses.
Variant R : network outputs the 5 differential states, algebraic states recovered exactly (x_a = -A22^{-1} A21 x_d).
Boundary conditions: hard (exact by construction) or soft (weight 100). Budget: N gradient evaluations per run."""
import sys, json, time, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
import network_core as nc

VS, IS, T = nc.VSC, nc.ISC, nc.T
S = np.array([VS]*3 + [IS]*2 + [VS]*2)                 # state scales
RS = np.array([1/IS]*3 + [1/VS]*2 + [1/IS]*2)           # row scales (node rows: A, inductor rows: V, algebraic rows: A)
Ej, Aj, Bj = (jnp.array(a) for a in (nc.E, nc.A, nc.B))
Sj, RSj = jnp.array(S), jnp.array(RS)
MREC = jnp.array(nc.M_REC)
V0s = jnp.array(nc.V0/VS); ITs = jnp.array(nc.IT/IS)
PH, AMP = jnp.array(nc.PH), jnp.array(nc.AMP)
TAU_C = jnp.linspace(0.0, 1.0, 101)
tau_eval = np.linspace(0, 1, 501); REF = nc.reference(tau_eval*T)     # (7,501,3) physical

def init(key, w=64, d=4, nout=7):
    sizes = [1] + [w]*d + [nout*3]; ps = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key); ps.append((jax.random.normal(k, (b, a))*np.sqrt(2/(a+b)), jnp.zeros(b)))
    return ps
def raw(ps, tau, nout):
    h = jnp.atleast_1d(tau)
    for W, b in ps[:-1]: h = jnp.tanh(W@h + b)
    W, b = ps[-1]; return (W@h + b).reshape(nout, 3)
def state(ps, tau, variant, hard):
    nout = 7 if variant == 'D' else 5
    N = raw(ps, tau, nout)
    if hard:
        N0, N1 = raw(ps, 0.0, nout), raw(ps, 1.0, nout)
        v = N[:3] + (1 - tau)*(V0s - N0[:3])
        i = N[3:5] + tau*(ITs - N1[3:5])
        xd = jnp.vstack([v, i])
    else: xd = N[:5]
    if variant == 'D': return jnp.vstack([xd, N[5:7]])
    xa = (MREC @ (Sj[:5, None]*xd))/Sj[5:, None]            # exact algebraic recovery (scaled units)
    return jnp.vstack([xd, xa])
def sources(tau):
    u1 = AMP*(0.80 + 0.30*jnp.sin(2*jnp.pi*tau + PH)); u3 = 0.10 + 0.05*jnp.cos(2*jnp.pi*tau + PH)
    return jnp.vstack([u1, u3])
def residual(ps, tau, variant, hard):          # physical residual of E x' = A x + B u, shape (7,3)
    xs, dxs = jax.jvp(lambda s: state(ps, s, variant, hard), (tau,), (1.0,))
    return Ej @ (Sj[:, None]*dxs/T) - Aj @ (Sj[:, None]*xs) - Bj @ sources(tau)

CACHE = {}
def build(variant, hard):
    key = (variant, hard)
    if key in CACHE: return CACHE[key]
    nout = 7 if variant == 'D' else 5
    flat0, unr = ravel_pytree(init(jax.random.PRNGKey(0), nout=nout))
    def parts(f):
        ps = unr(f); r = jax.vmap(lambda t: residual(ps, t, variant, hard))(TAU_C)*RSj[None, :, None]
        ld = jnp.mean(r[:, :5]**2); la = jnp.mean(r[:, 5:]**2)
        lb = 0.0
        if not hard:
            x0 = state(ps, 0.0, variant, hard); x1 = state(ps, 1.0, variant, hard)
            lb = jnp.mean((x0[:3] - V0s)**2) + jnp.mean((x1[3:5] - ITs)**2)
        return ld, la, lb
    def loss(f):
        ld, la, lb = parts(f); return ld + la + 100.0*lb
    lg = jax.jit(jax.value_and_grad(loss))
    @jax.jit
    def step(f, m, v, k):
        l, g = jax.value_and_grad(loss)(f); m = .9*m + .1*g; v = .999*v + .001*g**2
        return f - 1e-3*(m/(1 - .9**k))/(jnp.sqrt(v/(1 - .999**k)) + 1e-8), m, v, l
    @jax.jit
    def evaluate(f):
        ps = unr(f); xs = jax.vmap(lambda t: state(ps, t, variant, hard))(jnp.asarray(tau_eval))
        r = jax.vmap(lambda t: residual(ps, t, variant, hard))(jnp.asarray(tau_eval))
        x0 = state(ps, 0.0, variant, hard); x1 = state(ps, 1.0, variant, hard)
        return xs, r, x0, x1, jnp.stack(parts(f))
    CACHE[key] = (unr, lg, step, evaluate, nout); return CACHE[key]

def run(variant, hard, mode, frac, seed, N=2000):
    unr, lg, step, evaluate, nout = build(variant, hard)
    flat = ravel_pytree(init(jax.random.PRNGKey(seed), nout=nout))[0]
    na = {'adam': N, 'lbfgs': 0, 'two': int(round(frac*N))}[mode]; nl = N - na
    t0 = time.time(); m = jnp.zeros_like(flat); v = jnp.zeros_like(flat)
    for k in range(1, na+1): flat, m, v, l = step(flat, m, v, float(k))
    nfev = 0
    if nl > 0:
        fun = lambda x: tuple(np.asarray(a, dtype=np.float64) for a in lg(jnp.asarray(x)))
        r = minimize(fun, np.asarray(flat), jac=True, method='L-BFGS-B', options=dict(maxfun=nl, maxiter=nl, maxcor=50, ftol=1e-18, gtol=1e-14))
        flat = jnp.asarray(r.x); nfev = int(r.nfev)
    sec = time.time() - t0
    xs, r, x0, x1, pr = evaluate(flat); xs = np.asarray(xs); r = np.asarray(r)
    X = np.transpose(xs, (1, 0, 2))*S[:, None, None]        # (7,501,3) physical
    e = X - REF; rel = lambda a, b: float(np.linalg.norm(a)/np.linalg.norm(b))
    # constraint violation of the algebraic relation (A21 x_d + A22 x_a), independent of the loss
    viol = np.abs(np.einsum('ij,jtc->itc', nc.A21, X[:5]) + np.einsum('ij,jtc->itc', nc.A22, X[5:])).max()
    rphys = np.transpose(r, (1, 0, 2))                       # (7,501,3)
    x0p = np.asarray(x0)*S[:, None]; x1p = np.asarray(x1)*S[:, None]
    out = dict(variant=variant, hard=hard, mode=mode, frac=(1.0 if mode == 'adam' else 0.0 if mode == 'lbfgs' else frac), seed=seed,
        relL2_total=rel(e, REF), relL2_v=rel(e[:3], REF[:3]), relL2_i=rel(e[3:5], REF[3:5]), relL2_alg=rel(e[5:], REF[5:]),
        relL2_comp=[rel(e[k], REF[k]) for k in range(7)],
        maxabs_v=float(np.abs(e[:3]).max()), maxabs_i=float(np.abs(e[3:5]).max()), maxabs_alg=float(np.abs(e[5:]).max()),
        diff_res_rms=float(np.sqrt(np.mean((rphys[:5]*RS[:5, None, None])**2))), alg_res_rms=float(np.sqrt(np.mean((rphys[5:]*RS[5:, None, None])**2))),
        alg_constraint_violation_max=float(viol), bc_v_max=float(np.abs(x0p[:3] - nc.V0).max()), bc_i_max=float(np.abs(x1p[3:5] - nc.IT).max()),
        loss_diff=float(pr[0]), loss_alg=float(pr[1]), loss_bc=float(pr[2]), adam_evals=na, lbfgs_nfev=nfev, sec=sec)
    out['fail'] = bool(not np.isfinite(out['relL2_total']) or out['relL2_total'] > 5e-2)
    return out

if __name__ == "__main__":
    s0, s1 = int(sys.argv[1]), int(sys.argv[2]); fn = sys.argv[3] if len(sys.argv) > 3 else 'network_pinn_results.jsonl'
    done = set()
    try:
        for l in open(fn):
            r = json.loads(l); done.add((r['variant'], r['hard'], r['mode'], r['frac'], r['seed']))
    except FileNotFoundError: pass
    hard_cfg = [('adam', 1.0), ('lbfgs', 0.0), ('two', 0.1), ('two', 0.3), ('two', 0.5)]
    soft_cfg = [('adam', 1.0), ('lbfgs', 0.0), ('two', 0.3)]
    for variant, hard in (('D', True), ('R', True), ('D', False), ('R', False)):
        for md, fr in (hard_cfg if hard else soft_cfg):
            build(variant, hard)
    for seed in range(s0, s1):
        for hard, cfg in ((True, hard_cfg), (False, soft_cfg)):
            for variant in ('D', 'R'):
                for mode, fr in cfg:
                    if (variant, hard, mode, fr, seed) in done: continue
                    r = run(variant, hard, mode, fr, seed)
                    with open(fn, 'a') as f: f.write(json.dumps(r) + '\n')
                    print(variant, 'hard' if hard else 'soft', mode, fr, seed, f"{r['relL2_total']:.2e}", f"{r['sec']:.1f}s", flush=True)
