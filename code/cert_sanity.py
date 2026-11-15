import numpy as np, jax, jax.numpy as jnp, cert, network_pinn as npn, network_core as nc
unr, flat, evaluate = cert.train('R', 0)
ps = [(np.asarray(W), np.asarray(b)) for W, b in unr(flat)]
c = cert.certify(ps, 'R'); print('Kd', cert.CONST['Kd'], 'sup_g enclosure', c['sup_g'], 'sup_rd', c['sup_rd'])
# dense sampled sup of residual and of g
taus = jnp.linspace(0, 1, 40001)
rr = np.asarray(jax.vmap(lambda t: npn.residual(unr(flat), t, 'R', True))(taus))      # (40001,7,3)
G = cert.CONST['G']
g = np.einsum('gp,tpc->tgc', G, rr); print('sampled sup ||g||', np.sqrt((g**2).sum(1)).max(0), ' sampled sup ||r_d||', np.sqrt((rr[:, :5]**2).sum(1)).max(0), 'sup ||r_a||', np.sqrt((rr[:, 5:]**2).sum(1)).max(0))
# test of the solution-operator constant: random smooth g, solve e_d' = Mred e_d + g with e_v(0)=0, e_i(T)=0, compare ||e||/||g||
from scipy.integrate import solve_ivp
T = nc.T; rng = np.random.default_rng(0); worst = 0
for trial in range(30):
    a = rng.normal(size=(5, 4)); f = lambda t: a @ np.cos(np.pi*np.arange(4)*t/T*rng.uniform(1, 6)) if False else None
    freqs = rng.uniform(0, 3000, size=(5,)); ph = rng.uniform(0, 6.3, size=5); amp = rng.normal(size=5)
    gfun = lambda t: amp*np.sin(freqs*t + ph)
    opt = dict(rtol=1e-11, atol=1e-13, method='DOP853', dense_output=True)
    xp = solve_ivp(lambda t, x: nc.Mred @ x + gfun(t), (0, T), np.zeros(5), **opt)
    Phi = nc.Phi; z = np.linalg.solve(Phi[3:5, 3:5], -xp.sol(T)[3:5])
    sol = solve_ivp(lambda t, x: nc.Mred @ x + gfun(t), (0, T), np.r_[0, 0, 0, z], **opt)
    tt = np.linspace(0, T, 801); ed = sol.sol(tt); gg = np.array([gfun(t) for t in tt])
    worst = max(worst, np.linalg.norm(ed, axis=0).max()/np.linalg.norm(gg, axis=1).max())
print('worst observed ||e_d||/||g|| over 30 oscillatory g:', worst, ' vs Kd', cert.CONST['Kd'])
