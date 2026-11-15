"""Verification script for the proof package (Theorems 1-4).
Symbolic checks: sympy.  High-precision numerics: mpmath.  Independent check of Theorem 2: Chebyshev collocation.
Run:  python3 proof_verification.py   (prints PASS/FAIL lines, writes nothing else)"""
import numpy as np, sympy as sp, mpmath as mp, itertools, sys
from scipy.integrate import solve_ivp
mp.mp.dps = 30
np.set_printoptions(precision=6, suppress=True, linewidth=150)
RES = []
KINDS = []
KIND_BY_PREFIX = [("V1 ", "symbolic identity"), ("V2 ", "numerical identity"), ("V3 ", "symbolic counterexample"),
    ("V4", "independent numerical (collocation)"), ("V5", "high-precision evaluation"), ("V6", "independent numerical / symbolic"),
    ("V7", "numerical identity"), ("V8", "numerical identity"), ("V9a", "sampled sanity check"), ("V9b", "sampled sanity check (constants certified; comparison on a finite grid)"),
    ("V9c", "sampled sanity check (analytic inequality, floating-point evaluation)"), ("V9d", "source/script consistency"),
    ("V10 Example 2", "bookkeeping (restates Theorem 3(d); not an independent check)"), ("V10", "numerical rank computation"), ("V11", "independent symbolic / numerical")]
def check(name, ok, info=""):
    kind = next((k for p, k in KIND_BY_PREFIX if name.startswith(p)), "other")
    RES.append(ok); KINDS.append(kind)
    print(("PASS  " if ok else "FAIL  ") + f"[{kind}] " + name + ("  | " + info if info else ""))

# ---------------------------------------------------------------- V1  Schur / block determinant identity
s = sp.symbols('s')
for (r, m) in [(1, 1), (2, 1), (1, 2), (2, 2)]:
    A11 = sp.Matrix(r, r, lambda i, j: sp.Symbol(f'a{i}{j}')); A12 = sp.Matrix(r, m, lambda i, j: sp.Symbol(f'b{i}{j}'))
    A21 = sp.Matrix(m, r, lambda i, j: sp.Symbol(f'c{i}{j}')); A22 = sp.Matrix(m, m, lambda i, j: sp.Symbol(f'd{i}{j}'))
    n = r + m
    D = sp.diag(sp.eye(r), sp.zeros(m, m)); At = sp.Matrix(sp.BlockMatrix([[A11, A12], [A21, A22]]))
    lhs = (s * D - At).det()
    Ared = A11 - A12 * A22.inv() * A21
    rhs = (-A22).det() * (s * sp.eye(r) - Ared).det()
    check(f"V1 det(sD-A)=det(-A22)det(sI-Ared)  (r={r}, n-r={m}) symbolic", sp.simplify(lhs - rhs) == 0)

# ---------------------------------------------------------------- V2  basis independence of U^T A V ; regular pencil is not index 1
rng = np.random.default_rng(1)
n, r = 5, 3
E = rng.standard_normal((n, r)) @ rng.standard_normal((r, n)); A = rng.standard_normal((n, n))
from scipy.linalg import null_space
V = null_space(E); U = null_space(E.T)
S_ = rng.standard_normal((n - r, n - r)); T_ = rng.standard_normal((n - r, n - r))
d1 = np.linalg.det(U.T @ A @ V); d2 = np.linalg.det((U @ S_).T @ A @ (V @ T_))
check("V2 det(U^T A V) scales by det(S)det(T): invertibility is basis-free", abs(d2 - np.linalg.det(S_) * np.linalg.det(T_) * d1) < 1e-9 * max(1, abs(d2)))
E2 = sp.Matrix([[0, 1], [0, 0]]); A2 = sp.eye(2)
check("V3 regular pencil but NOT index 1:  det(sE-A) = %s, U^T A V = %s" % ((s * E2 - A2).det(), (sp.Matrix([[0, 1]]) * A2 * sp.Matrix([1, 0]))[0]),
      (s * E2 - A2).det() == 1 and (sp.Matrix([[0, 1]]) * A2 * sp.Matrix([1, 0]))[0] == 0)

# ---------------------------------------------------------------- V4  Theorem 2 against independent Chebyshev collocation of the FULL descriptor system
def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N); c = np.r_[2, np.ones(N - 1), 2] * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T; dX = X - X.T
    Dm = (np.outer(c, 1 / c)) / (dX + np.eye(N + 1)); Dm -= np.diag(Dm.sum(1)); return Dm, x
def build_system(seed, n=5, r=3, k=4):
    rg = np.random.default_rng(seed)
    P_ = rg.standard_normal((n, n)); Q_ = rg.standard_normal((n, n))
    Dd = np.diag([1.] * r + [0.] * (n - r)); Ecs = np.linalg.inv(P_) @ Dd @ np.linalg.inv(Q_)   # P E Q = D
    Ac = [rg.standard_normal((n, n)) * 0.7 for _ in range(3)]
    # make the algebraic block invertible for all t:  A22(t)= -I*(2+sin t) + small
    def Afun(t):
        At = Ac[0] + Ac[1] * np.sin(t) + Ac[2] * t
        Tt = P_ @ At @ Q_; Tt[r:, r:] = -(2 + np.sin(t)) * np.eye(n - r) + 0.3 * Tt[r:, r:] * 0 + 0.2 * np.cos(t); return np.linalg.inv(P_) @ Tt @ np.linalg.inv(Q_)
    Afun.Ac = Ac
    fv = rg.standard_normal(n)
    ffun = lambda t: fv * np.cos(2 * t) + np.sin(t)
    return Ecs, Afun, ffun, P_, Q_, Dd
def theory(Ecs, Afun, ffun, P_, Q_, r, L0, L1, w):
    n = Ecs.shape[0]; Qi = np.linalg.inv(Q_)
    def blocks(t):
        Tt = P_ @ Afun(t) @ Q_; return Tt[:r, :r], Tt[:r, r:], Tt[r:, :r], Tt[r:, r:]
    def red(t):
        A11, A12, A21, A22 = blocks(t); ft = P_ @ ffun(t); Ai = np.linalg.inv(A22)
        return A11 - A12 @ Ai @ A21, ft[:r] - A12 @ Ai @ ft[r:]
    def rhs(t, z):  # z = [Phi (r*r) ; yp (r)]
        Ar, fr = red(t); Phi = z[:r * r].reshape(r, r); return np.r_[(Ar @ Phi).ravel(), Ar @ z[r * r:] + fr]
    z0 = np.r_[np.eye(r).ravel(), np.zeros(r)]
    sol = solve_ivp(rhs, (0, 1), z0, rtol=1e-13, atol=1e-14, method='DOP853')
    Phi1 = sol.y[:r * r, -1].reshape(r, r); yp1 = sol.y[r * r:, -1]
    def Theta_and_part(t, Phi, yp):
        A11, A12, A21, A22 = blocks(t); Ai = np.linalg.inv(A22); ft = P_ @ ffun(t)
        Th = np.vstack([np.eye(r), -Ai @ A21]); ya = -Ai @ (A21 @ yp + ft[r:])
        return Q_ @ Th @ Phi, Q_ @ np.r_[yp, ya]
    Psi0, xp0 = Theta_and_part(0.0, np.eye(r), np.zeros(r)); Psi1, xp1 = Theta_and_part(1.0, Phi1, yp1)
    M = L0 @ Psi0 + L1 @ Psi1; b = w - L0 @ xp0 - L1 @ xp1
    return M, b
def collocate(Ecs, Afun, ffun, L0, L1, N=36):
    n = Ecs.shape[0]; Dm, x = cheb(N); t = (x + 1) / 2; Dt = 2 * Dm
    K = np.kron(Dt, Ecs)                              # E x'
    for i in range(N + 1): K[i * n:(i + 1) * n, i * n:(i + 1) * n] -= Afun(t[i])
    rhs = np.concatenate([ffun(t[i]) for i in range(N + 1)])
    ix0 = N * n; ix1 = 0   # x=cos(pi j/N): j=0 -> t=1 ; j=N -> t=0
    Bm = np.zeros((L0.shape[0], (N + 1) * n)); Bm[:, N * n:(N + 1) * n] = L0; Bm[:, 0:n] = L1
    return K, rhs, Bm, t
n, r, k = 5, 3, 4
ok_all = True; used = 0
for seed in range(40):
    if used == 5: break
    Ecs, Afun, ffun, P_, Q_, Dd = build_system(seed, n, r, k)
    rg = np.random.default_rng(100 + seed)
    L0 = rg.standard_normal((k, n)); L1 = rg.standard_normal((k, n)); wtrue = rg.standard_normal(k)
    M, b = theory(Ecs, Afun, ffun, P_, Q_, r, L0, L1, wtrue)
    if np.linalg.cond(M) > 1e6 or np.linalg.norm(b) > 1e6:      # skip systems whose fundamental matrix grows like e^{40}: round-off, not theory
        print(f"   seed {seed}: skipped (cond(M)={np.linalg.cond(M):.1e}, ill-conditioned test system)"); continue
    used += 1
    rkM = np.linalg.matrix_rank(M, tol=1e-9); defect = np.linalg.norm(b - M @ np.linalg.lstsq(M, b, rcond=None)[0])
    K, rhs, Bm, t = collocate(Ecs, Afun, ffun, L0, L1, N=48)
    ns = null_space(K, rcond=1e-10); zp = np.linalg.lstsq(K, rhs, rcond=None)[0]
    Mc = Bm @ ns; bc = wtrue - Bm @ zp; dcoll = np.linalg.norm(bc - Mc @ np.linalg.lstsq(Mc, bc, rcond=None)[0])
    ok = (ns.shape[1] == r) and (rkM == np.linalg.matrix_rank(Mc, tol=1e-8)) and abs(defect - dcoll) < 1e-6 * max(1, defect)
    ok_all &= ok
    print(f"   seed {seed}: dim ker K(coll)={ns.shape[1]} (r={r}); rank M theory/colloc={rkM}/{np.linalg.matrix_rank(Mc, tol=1e-8)}; defect theory={defect:.8f} colloc={dcoll:.8f}")
    alpha = rg.standard_normal(r); M0, b0 = theory(Ecs, Afun, ffun, P_, Q_, r, L0, L1, np.zeros(k))
    wcons = (-b0) + M @ alpha
    Mx, bx = theory(Ecs, Afun, ffun, P_, Q_, r, L0, L1, wcons)
    ok_all &= np.linalg.norm(bx - Mx @ alpha) < 1e-8
ok_all &= (used == 5)
check("V4 Theorem 2: rank M, kernel dimension r, compatibility defect agree with independent Chebyshev collocation of the full descriptor system", ok_all)
# (c) rank-deficient boundary operator: nullity of homogeneous BVP = r - rank M
Ecs, Afun, ffun, P_, Q_, Dd = build_system(7, n, r, 2)
rg = np.random.default_rng(5); L0 = rg.standard_normal((2, n)); L1 = rg.standard_normal((2, n))   # k=2<r=3 -> nullity >= 1
M, b = theory(Ecs, Afun, ffun, P_, Q_, r, L0, L1, np.zeros(2)); K, rhs, Bm, t = collocate(Ecs, Afun, ffun, L0, L1)
ns = null_space(K, rcond=1e-10); nul_coll = ns.shape[1] - np.linalg.matrix_rank(Bm @ ns, tol=1e-8)
check("V4c homogeneous BVP solution-space dimension = r - rank M", nul_coll == r - np.linalg.matrix_rank(M, tol=1e-9), f"theory {r-np.linalg.matrix_rank(M,tol=1e-9)}, colloc {nul_coll}")

# ---------------------------------------------------------------- V5  the four original examples (high precision)
Fcols = {1: [lambda t: (t, mp.cos(t)), lambda t: (mp.e ** t, t ** 2)],
         2: [lambda t: (0, mp.e ** (2 * t)), lambda t: (t * mp.sin(t), mp.cos(2 * t))],
         3: [lambda t: (mp.sin(t), 0), lambda t: (mp.e ** t, 0)],
         4: [lambda t: (mp.cos(t), 0), lambda t: (t, 0)]}
Aconst = {1: [[1, 2], [3, 4]], 3: [[0, 1], [1, 1]], 4: [[1, 1], [2, 2]]}
L1d = {1: (-2, 1), 2: (-2, 1), 3: (-1, 1), 4: (-1, 1)}
Wd = {1: [[0.6887, -0.9577], [4.5984, 7.6409]], 2: [[39.6326, 7.6130], [-10.8047, -4.2308]],
      3: [[-0.6506, -0.8591], [0.3494, -1.8591]], 4: [[-0.8415, -0.5], [-0.8415, -0.5]]}
print("\n   Example  col | A22  A_red | rank M | M^T = (BC sensitivities to alpha) | compat. defect of rounded W | alpha_LS | max |W - W*(alpha_LS)|")
defects = {}; Mstore = {}
for ex in (1, 3, 4):
    A = [[mp.mpf(v) for v in row] for row in Aconst[ex]]; A11, A12, A21, A22 = A[0][0], A[0][1], A[1][0], A[1][1]
    Ar = A11 - A12 * A21 / A22
    for kc in range(2):
        f = Fcols[ex][kc]
        fred = lambda t: f(t)[0] - A12 * f(t)[1] / A22
        yp = lambda t: mp.quad(lambda s_: mp.e ** (Ar * (t - s_)) * fred(s_), [0, t]) if t > 0 else mp.mpf(0)
        xcomp = lambda y, t: (y, -(A21 * y + f(t)[1]) / A22)
        psi = lambda t: (mp.e ** (Ar * t), -A21 / A22 * mp.e ** (Ar * t))
        L1m = L1d[ex]
        Mv = [psi(0)[i] + L1m[i] * psi(1)[i] for i in range(2)]
        x0 = xcomp(mp.mpf(0), mp.mpf(0)); x1 = xcomp(yp(mp.mpf(1)), mp.mpf(1))
        bv = [mp.mpf(Wd[ex][i][kc]) - (x0[i] + L1m[i] * x1[i]) for i in range(2)]
        Mn = np.array([float(v) for v in Mv]); bn = np.array([float(v) for v in bv])
        a = float(np.dot(Mn, bn) / np.dot(Mn, Mn)); res = bn - Mn * a
        defects[(ex, kc)] = np.linalg.norm(res); Mstore[(ex, kc)] = Mn.copy()
        print(f"   Ex{ex}     {kc+1}  | {float(A22):.0f}  {float(Ar):+.2f} | 1      | {Mn} | {np.linalg.norm(res):.3e} (rows {res}) | {a:+.6f} | {np.max(np.abs(res)):.2e}")
check("V5 Examples 1,3,4: computed rank M = 1 (= r) in every column, k = 2 rows -> exactly one compatibility condition per column", all(np.linalg.matrix_rank(v.reshape(2, 1)) == 1 for v in Mstore.values()))
check("V5 Example 4: computed M = (0, -2): row 1 has zero sensitivity to alpha (pure data-consistency condition)", all(abs(Mstore[(4, c)][0]) < 1e-12 and abs(Mstore[(4, c)][1] + 2) < 1e-12 for c in (0, 1)))
print("      Example 4 exact consistent data: W* = [[-sin 1, -1/2],[-sin 1,-1/2]] =", [[-float(mp.sin(1)), -0.5]] * 2, " vs given [[-0.8415,-0.5],[-0.8415,-0.5]]  (diff col1 = %.2e)" % (float(mp.sin(1)) - 0.8415))
# Example 2 -------------------------------------------------------
print()
def ex2_col(kc):
    f = Fcols[2][kc]; f1 = lambda t: f(t)[0]; f2 = lambda t: f(t)[1]
    a = lambda t: mp.cos(t) - mp.cos(3 * t) / mp.cos(t); bq = lambda t: f1(t) - f2(t) / mp.cos(t)
    c = -f2(mp.mpf(0))
    sol = mp.odefun(lambda t, y: a(t) * y + bq(t), 0, c)
    x2 = lambda t: sol(t); x1 = lambda t: -(mp.cos(3 * t) * x2(t) + f2(t)) / mp.sin(t)
    df2 = mp.diff(f2, 0); x10 = f2(0) - f1(0) - df2
    return f1, f2, c, x1, x2, x10
Wstar = np.zeros((2, 2))
for kc in range(2):
    f1, f2, c, x1, x2, x10 = ex2_col(kc)
    lim = x1(mp.mpf('1e-8')); 
    Wstar[0, kc] = float(x10 - 2 * x1(mp.mpf(1))); Wstar[1, kc] = float(x2(0) + x2(1))
    check(f"V6 Example 2 col {kc+1}: x2(0)=-f2(0)={float(c):.0f}; x1(0)=f2(0)-f1(0)-f2'(0)={float(x10):+.0f} equals numerical limit", abs(lim - x10) < 1e-6, f"x1(1e-8)={float(lim):+.8f}")
    # blow-up if c != -f2(0):  t*x1(t) -> -(c+f2(0))
    cc = c + mp.mpf('0.1'); sol2 = mp.odefun(lambda t, y: (mp.cos(t) - mp.cos(3 * t) / mp.cos(t)) * y + f1(t) - f2(t) / mp.cos(t), 0, cc)
    tt = mp.mpf('1e-6'); lim2 = tt * (-(mp.cos(3 * tt) * sol2(tt) + f2(tt)) / mp.sin(tt))
    check(f"V6 Example 2 col {kc+1}: lim t*x1(t) = -(x2(0)+f2(0)) when x2(0)!=-f2(0) (x1 ~ 1/t)", abs(lim2 + mp.mpf('0.1')) < 1e-5, f"{float(lim2):+.6f} (expected -0.1)")
for kc in range(2):
    dd = np.array(Wd[2])[:, kc] - Wstar[:, kc]; print(f"      Example 2 col {kc+1}: W - W* = {dd}, norm {np.linalg.norm(dd):.2e}")
f1_, f2_, c_, x1_, x2_, x10_ = ex2_col(0); h_ = mp.mpf('1e-6'); d1 = (x1_(h_) - x10_) / h_
check("V6 x1'(0) for col 1 equals symbolic formula -p1/2-9q0/2+q1/2-q2/2 = -11/2", abs(d1 + mp.mpf(11) / 2) < 1e-4, f"numeric {float(d1):.6f}")
print("      Example 2: W*(bounded solution) =\n", Wstar, "\n      given W =\n", np.array(Wd[2]), "\n      |W - W*| =", np.abs(np.array(Wd[2]) - Wstar).max())
# symbolic endpoint expansion with generic Taylor data
t = sp.symbols('t'); p = sp.symbols('p0:5'); q = sp.symbols('q0:5'); c0 = -q[0]
f1s = sum(p_ * t ** i / sp.factorial(i) for i, p_ in enumerate(p[:4])) ; f2s = sum(q_ * t ** i / sp.factorial(i) for i, q_ in enumerate(q[:4]))
# f1 -> coefficients p (f1^(i)(0)=p_i),  f2 -> q
f1s = sum(p[i] * t ** i / sp.factorial(i) for i in range(4)); f2s = sum(q[i] * t ** i / sp.factorial(i) for i in range(4))
a_s = sp.cos(t) - sp.cos(3 * t) / sp.cos(t); b_s = f1s - f2s / sp.cos(t)
x2s = c0 + sum(sp.Symbol(f'u{i}') * t ** i for i in range(1, 5))
eqn = sp.series(sp.diff(x2s, t) - a_s * x2s - b_s, t, 0, 4).removeO()
sol_u = sp.solve([eqn.coeff(t, i) for i in range(4)], [sp.Symbol(f'u{i}') for i in range(1, 5)], dict=True)[0]
x2s = x2s.subs(sol_u)
g = -(sp.cos(3 * t) * x2s + f2s); x1s = sp.series(g / sp.sin(t), t, 0, 3).removeO()
x10s = sp.simplify(x1s.coeff(t, 0)); x11s = sp.simplify(x1s.coeff(t, 1))
check("V6 symbolic: x1(0) = q0... i.e. f2(0)-f1(0)-f2'(0)", sp.simplify(x10s - (q[0] - p[0] - q[1])) == 0, f"x1(0)={x10s}")
print("      symbolic: x2(t) = ", sp.simplify(x2s).series(t, 0, 3).removeO(), "\n      symbolic: x1'(0) =", x11s)
# ---------------------------------------------------------------- V7  hard BC construction
k_, n_ = 3, 4
rg = np.random.default_rng(11)
L0 = rg.standard_normal((k_, n_)); L1 = rg.standard_normal((k_, n_)); w = rg.standard_normal(k_)
B = np.hstack([L0, L1]); H = np.linalg.pinv(B); H0, H1 = H[:n_], H[n_:]
check("V7 B H = I_k (right inverse exists iff rank B = k)", np.allclose(B @ H, np.eye(k_)) and np.linalg.matrix_rank(B) == k_)
Nf = lambda t: np.array([np.sin(3 * t + i) + t ** i for i in range(n_)])
def xN(t): return Nf(t) + ((1 - t) * H0 + t * H1) @ (w - L0 @ Nf(0) - L1 @ Nf(1))
check("V7 hard-BC map gives L0 x(0)+L1 x(1) = w for an arbitrary N", np.allclose(L0 @ xN(0) + L1 @ xN(1), w, atol=1e-13))
# idempotence: if N satisfies BC, x_N = N
Nb = lambda t: Nf(t) + ((1 - t) * H0 + t * H1) @ (w - L0 @ Nf(0) - L1 @ Nf(1))
xx = lambda t: Nb(t) + ((1 - t) * H0 + t * H1) @ (w - L0 @ Nb(0) - L1 @ Nb(1))
check("V7 projection property: x_N = N whenever N already satisfies the BC", all(np.allclose(xx(tt), Nb(tt)) for tt in (0, .3, .7, 1)))
# original paper formula, L0 = I
n2 = 2; L1m = np.diag([-2., 1.]); w2 = np.array([0.3, -0.7]); N2 = lambda t: np.array([np.sin(t) + 1, np.cos(2 * t)])
x_orig = lambda t: N2(t) + (1 - t) * (w2 - N2(0) - L1m @ N2(1))
check("V7 paper formula (L0=I): X=N+(1-t)[W-N(0)-L1 N(1)] satisfies BC; H0=I,H1=0", np.allclose(x_orig(0) + L1m @ x_orig(1), w2))
# rank-deficient B (a duplicated row): no right inverse
Bd = np.vstack([B[:2], B[:1]]); check("V7 duplicated BC row: rank B < k -> no right inverse (construction cannot exist; data must be consistent)", np.linalg.matrix_rank(Bd) == 2 < 3)
# ---------------------------------------------------------------- V8  consistency-floor identity (Proposition 3)
Ecs, Afun, ffun, P_, Q_, Dd = build_system(2, 5, 3, 4); rg = np.random.default_rng(3)
L0 = rg.standard_normal((4, 5)); L1 = rg.standard_normal((4, 5)); w = rg.standard_normal(4)
M, b = theory(Ecs, Afun, ffun, P_, Q_, 3, L0, L1, w); Pr = np.eye(4) - M @ np.linalg.pinv(M)
# a BC-satisfying smooth function x (hard BC applied to N), its residual r(t) = E x' - A x - f
H = np.linalg.pinv(np.hstack([L0, L1])); H0, H1 = H[:5], H[5:]
Nf = lambda t: np.array([np.sin(2 * t + i) * (1 + t) for i in range(5)]); dN = lambda t: np.array([2 * np.cos(2 * t + i) * (1 + t) + np.sin(2 * t + i) for i in range(5)])
cc_ = w - L0 @ Nf(0) - L1 @ Nf(1)
xf = lambda t: Nf(t) + ((1 - t) * H0 + t * H1) @ cc_; dxf = lambda t: dN(t) + (H1 - H0) @ cc_
rfun = lambda t: Ecs @ dxf(t) - Afun(t) @ xf(t) - ffun(t)
# x solves the problem with forcing f + r; so (I - M M+) b(f + r) = 0 ; and b(f+r) - b(f) = -(L0 xp[r](0)+L1 xp[r](1))
Mr, br = theory(Ecs, Afun, lambda t: ffun(t) + rfun(t), P_, Q_, 3, L0, L1, w)
check("V8 Proposition 3 identity: x BC-feasible & residual r  =>  (I-MM^+) b(f+r) = 0 ; hence defect(f) = |(I-MM^+)(L0 xp[r](0)+L1 xp[r](1))|",
      np.linalg.norm(Pr @ br) < 1e-7, f"|P b(f+r)|={np.linalg.norm(Pr@br):.2e}; |P b(f)|={np.linalg.norm(Pr@b):.4f}")

# ---------------------------------------------------------------- V9  a-posteriori error bound (Theorem 5), explicit constants
print()
def apost_constants(blocksfun, Phi_gamma, Psi_sup, Mpinv_norm, L0, L1, P_, Q_, r):
    """explicit constants of Theorem 5 from sup-bounds of the data (evaluated on a fine grid)"""
    tg = np.linspace(0, 1, 401); n = Q_.shape[0]
    a12 = max(np.linalg.norm(b_[1] @ np.linalg.inv(b_[3]), 2) for b_ in map(blocksfun, tg))
    a21 = max(np.linalg.norm(b_[2], 2) for b_ in map(blocksfun, tg)); a22 = max(np.linalg.norm(np.linalg.inv(b_[3]), 2) for b_ in map(blocksfun, tg))
    cg = 1 + a12; Ce = np.linalg.norm(Q_, 2) * (Phi_gamma * cg * (1 + a22 * a21) + a22)
    Cb = Psi_sup * Mpinv_norm
    Crho = Ce * (1 + Cb * (np.linalg.norm(L0, 2) + np.linalg.norm(L1, 2)))
    return Cb, Crho * np.linalg.norm(P_, 2)
def sup_t(f, ng=801): return max(np.linalg.norm(f(t_)) for t_ in np.linspace(0, 1, ng))
ok9 = True; rows = []
for ex in (1, 3, 4):
    Am = np.array(Aconst[ex], float); Ec = np.diag([1., 0.]); L1m = np.diag(np.array(L1d[ex], float)); L0m = np.eye(2)
    Ar = Am[0, 0] - Am[0, 1] * Am[1, 0] / Am[1, 1]; col = np.array([1., -Am[1, 0] / Am[1, 1]])
    Psi = lambda t_: (col * np.exp(Ar * t_)).reshape(2, 1)
    M = L0m @ Psi(0) + L1m @ Psi(1); Mp = np.linalg.pinv(M); gam = max(1., np.exp(Ar))
    bl = lambda t_: (Am[:1, :1], Am[:1, 1:], Am[1:, :1], Am[1:, 1:])
    Cb, Cr = apost_constants(bl, gam, sup_t(Psi), np.linalg.norm(Mp, 2), L0m, L1m, np.eye(2), np.eye(2), 1)
    for eps in (1e-1, 1e-3):
        eta = lambda t_: eps * np.array([np.sin(3 * t_) + t_ ** 2, np.cos(2 * t_) - 0.5 * t_])
        deta = lambda t_: eps * np.array([3 * np.cos(3 * t_) + 2 * t_, -2 * np.sin(2 * t_) - 0.5])
        res = lambda t_: Ec @ deta(t_) - Am @ eta(t_)               # x_hat = x + eta (any exact x): residual of eta in the homogeneous equation
        beta = L0m @ eta(0.) + L1m @ eta(1.)
        # alpha-part: the bound is for the error e = eta *when beta = L0 e(0)+L1 e(1)*; but rank M=1<k=2 means eta must be a perturbation of a consistent problem:
        # use the generalised statement with the least-squares alpha (Theorem 5 (b)): |e|_inf <= Cb |beta| + Crho |P r|_inf
        lhs = sup_t(eta); rhs = Cb * np.linalg.norm(beta) + Cr * sup_t(res)
        ok9 &= lhs <= rhs; rows.append((ex, eps, lhs, rhs))
print("   Theorem 5 check (Examples 1,3,4, arbitrary perturbation eta): |e|_inf <= C_beta|beta| + C_rho |r|_inf")
for ex, eps, l, r_ in rows: print(f"      Ex{ex} eps={eps:g}: |e|_inf={l:.3e} <= bound={r_:.3e}  (ratio {r_/l:.1f})")
check("V9a a-posteriori bound holds on Examples 1,3,4 (constants from analytic formulas, floating point)", ok9)
# random index-1 systems
ok9b = True; cnt = 0
for seed in range(40):
    if cnt == 5: break
    Ecs, Afun, ffun, P_, Q_, Dd = build_system(seed, 5, 3, 4); rg = np.random.default_rng(500 + seed)
    L0 = rg.standard_normal((4, 5)); L1 = rg.standard_normal((4, 5))
    M, b = theory(Ecs, Afun, ffun, P_, Q_, 3, L0, L1, np.zeros(4))
    if np.linalg.cond(M) > 1e5:
        print(f"      random system seed {seed}: SKIPPED (cond(M)={np.linalg.cond(M):.1e} > 1e5, ill-conditioned test system)"); continue
    cnt += 1
    Mp = np.linalg.pinv(M)
    bl = lambda t_: (lambda T: (T[:3, :3], T[:3, 3:], T[3:, :3], T[3:, 3:]))(P_ @ Afun(t_) @ Q_)
    # gamma = sup |Phi(t)Phi(s)^-1| ; Psi
    Ared = lambda t_: (lambda B_: B_[0] - B_[1] @ np.linalg.inv(B_[3]) @ B_[2])(bl(t_))
    sol = solve_ivp(lambda t_, z: (Ared(t_) @ z.reshape(3, 3)).ravel(), (0, 1), np.eye(3).ravel(), rtol=1e-12, atol=1e-13, dense_output=True)
    Phis = lambda t_: sol.sol(t_).reshape(3, 3)
    tg = np.linspace(0, 1, 41); gam = max(np.linalg.norm(Phis(t_) @ np.linalg.inv(Phis(s_)), 2) for t_ in tg for s_ in tg if s_ <= t_)
    def Psi(t_):
        B_ = bl(t_); Th = np.vstack([np.eye(3), -np.linalg.inv(B_[3]) @ B_[2]]); return Q_ @ Th @ Phis(t_)
    Cb, Cr = apost_constants(bl, gam, max(np.linalg.norm(Psi(t_), 2) for t_ in tg), np.linalg.norm(Mp, 2), L0, L1, P_, Q_, 3)
    # exact solution x (alpha random, consistent w), perturbation eta; e = eta
    alpha = rg.standard_normal(3); M0, b0 = M, b      # b computed with w=0
    eta = lambda t_: 1e-2 * np.array([np.sin(2 * t_ + i) + t_ * i for i in range(5)]); deta = lambda t_: 1e-2 * np.array([2 * np.cos(2 * t_ + i) + i for i in range(5)])
    # data consistent with the exact solution: w* = M alpha + L0 xp(0)+L1 xp(1)  -> e = x_hat - x ; here we TEST the bound on e = eta by requiring beta = L0 eta(0)+L1 eta(1) in range(M)+... (hard version):
    beta = L0 @ eta(0.) + L1 @ eta(1.); rfun = lambda t_: Ecs @ deta(t_) - Afun(t_) @ eta(t_)
    # (Theorem 5 requires e = Psi alpha + e_p with M alpha = beta - L[rho]; the part of beta outside range M cannot occur for a true error. project:)
    Pm = np.eye(4) - M @ Mp
    bt = beta - 0  # use full beta; bound is conservative
    lhs = sup_t(eta); rhs = Cb * np.linalg.norm(beta) + Cr * max(np.linalg.norm(rfun(t_)) for t_ in np.linspace(0, 1, 401))
    ok9b &= lhs <= rhs
    print(f"      random system seed {seed}: |e|_inf={lhs:.3e} <= bound={rhs:.3e}")
check("V9a bound holds on random index-1 systems with GRID-ESTIMATED constants (NOT certified: finite-grid maxima are not guaranteed upper bounds)", ok9b and cnt == 5)

# ---------------------------------------------------------------- V9b  CERTIFIED constants for Examples 1,3,4 (interval arithmetic, mpmath.iv)
print()
iv = mp.iv; iv.dps = 30
cert_rows = []; ok9c = True
for ex in (1, 3, 4):
    a11, a12_, a21_, a22_ = [iv.mpf(v) for v in (Aconst[ex][0][0], Aconst[ex][0][1], Aconst[ex][1][0], Aconst[ex][1][1])]
    lam = a11 - a12_ * a21_ / a22_
    elam = iv.exp(lam); gam_iv = iv.mpf(max(mp.mpf(1), mp.mpf(elam.b)))          # sup_{0<=s<=t<=1} e^{lam (t-s)} = max(1, e^lam)  (upper endpoint)
    A12s = abs(a12_ / a22_); A21s = abs(a21_); A22s = abs(1 / a22_)
    colv = iv.sqrt(1 + (a21_ / a22_) ** 2)
    Psi_sup = gam_iv * colv                                                         # sup_t e^{lam t} |(1,-a21/a22)| <= max(1,e^lam) |(1,-a21/a22)|
    l1 = [iv.mpf(v) for v in L1d[ex]]
    M0 = 1 + l1[0] * elam; M1 = -(a21_ / a22_) * (1 + l1[1] * elam)
    Mnorm_lo = iv.sqrt(M0 ** 2 + M1 ** 2).a                                       # lower endpoint of |M|
    Mp = 1 / iv.mpf(Mnorm_lo)                                                       # ||M^+|| = 1/|M| (M is a column vector)
    Q2 = iv.mpf(1); P2 = iv.mpf(1)
    Ce = Q2 * (gam_iv * (1 + A12s) * (1 + A22s * A21s) + A22s)
    Cb = Psi_sup * Mp; nL = iv.mpf(1) + max(abs(l1[0]), abs(l1[1]))
    Crho = P2 * Ce * (1 + Cb * nL)
    cert_rows.append((ex, float(Cb.b), float(Crho.b)))
    # compare with the floating-point constants of V9a
import math
up = lambda x, d: math.ceil(x * 10 ** d) / 10 ** d                      # OUTWARD (upward) rounding: never below the interval upper endpoint
print("   Certified constants (interval arithmetic; upper endpoints, rounded UPWARD):  (ex, C_beta, C_rho)")
for row in cert_rows: print(f"      Ex{row[0]}:  C_beta <= {up(row[1], 4):.4f} (endpoint {row[1]:.9f}),  C_rho <= {up(row[2], 2):.2f} (endpoint {row[2]:.9f})")
CERT = {row[0]: (row[1], row[2]) for row in cert_rows}
ok9d = True
for ex, Cb_, Cr_ in cert_rows:
    Am = np.array(Aconst[ex], float); Ec = np.diag([1., 0.]); L1m = np.diag(np.array(L1d[ex], float)); L0m = np.eye(2)
    for eps in (1e-1, 1e-3, 1e-6):
        eta = lambda t_: eps * np.array([np.sin(3 * t_) + t_ ** 2, np.cos(2 * t_) - 0.5 * t_])
        deta = lambda t_: eps * np.array([3 * np.cos(3 * t_) + 2 * t_, -2 * np.sin(2 * t_) - 0.5])
        res = lambda t_: Ec @ deta(t_) - Am @ eta(t_); beta = L0m @ eta(0.) + L1m @ eta(1.)
        ok9d &= sup_t(eta) <= Cb_ * np.linalg.norm(beta) + Cr_ * sup_t(res)
check("V9b Theorem 5 with interval-certified constants on Examples 1,3,4, compared with test functions whose suprema are taken on a finite grid (sampled sanity check, NOT a continuous-domain certification)", ok9d)
# ---------------------------------------------------------------- V9c  analytic (Gronwall-type) constants for random systems
print()
def analytic_constants(Afun, P_, Q_, L0, L1, r, M):
    """Rigorous (Lemma 2) bounds.  Blocks A11,A12,A21 of P A(t) Q are affine in (s,tau)=(sin t,t) in the box [0,sin 1]x[0,1];
    the spectral norm is convex, so its supremum over the box is attained at a corner (exact for the box, hence an upper bound on the curve)."""
    n = P_.shape[0]; m = n - r; Ac = Afun.Ac
    T = [P_ @ a_ @ Q_ for a_ in Ac]
    corners = [(sv, tv) for sv in (0.0, np.sin(1.0)) for tv in (0.0, 1.0)]
    def blk_sup(sl_r, sl_c):
        return max(np.linalg.norm((T[0] + T[1] * sv + T[2] * tv)[sl_r, sl_c], 2) for sv, tv in corners)
    A11s = blk_sup(slice(0, r), slice(0, r)); A12s = blk_sup(slice(0, r), slice(r, n)); A21s = blk_sup(slice(r, n), slice(0, r))
    a22 = 1.0 / (2.0 - 0.2 * m)       # A22(t) = -(2+sin t) I + 0.2 cos t * ones : |eig| >= (2+sin t) - 0.2 m cos t >= 2 - 0.2 m
    assert 2.0 - 0.2 * m > 0
    a21 = A21s; a12 = A12s * a22; bred = A11s + A12s * a22 * A21s
    gam = np.exp(bred)                                                          # Gronwall: |Phi(t)Phi(s)^-1| <= exp(int |A_red|)
    Psi_sup = np.linalg.norm(Q_, 2) * (1 + a22 * a21) * gam
    Mp = np.linalg.norm(np.linalg.pinv(M), 2)                                   # NOT interval-certified (M from DOP853, rtol 1e-13)
    Ce = np.linalg.norm(Q_, 2) * (gam * (1 + a12) * (1 + a22 * a21) + a22)
    Cb = Psi_sup * Mp; Crho = np.linalg.norm(P_, 2) * Ce * (1 + Cb * (np.linalg.norm(L0, 2) + np.linalg.norm(L1, 2)))
    return Cb, Crho, dict(gamma=gam, a12=a12, a21=a21, a22=a22, bred=bred)
def build_benign(seed, n=5, r=3):
    """P = Q = I, E = diag(I_r,0), mild coefficients: the analytic constants are then of moderate size"""
    rg = np.random.default_rng(seed); Ac = [rg.standard_normal((n, n)) * 0.4 for _ in range(3)]; Dd = np.diag([1.] * r + [0.] * (n - r))
    def Afun(t):
        At = Ac[0] + Ac[1] * np.sin(t) + Ac[2] * t; At[r:, r:] = -(2 + np.sin(t)) * np.eye(n - r) + 0.2 * np.cos(t); return At
    Afun.Ac = Ac; fv = rg.standard_normal(n)
    return Dd, Afun, (lambda t: fv * np.cos(2 * t) + np.sin(t)), np.eye(n), np.eye(n), Dd
for label, builder, nsys in (("benign (P=Q=I, mild coefficients)", build_benign, 5), ("wild (random invertible P,Q, large coefficients)", build_system, 5)):
    ok_lab = True; cnt = 0; skipped = []; print(f"      -- {label}")
    for seed in range(60):
        if cnt == nsys: break
        Ecs, Afun, ffun, P_, Q_, Dd = builder(seed) if builder is build_benign else builder(seed, 5, 3, 4); rg = np.random.default_rng(500 + seed)
        L0 = rg.standard_normal((4, 5)); L1 = rg.standard_normal((4, 5))
        M, b = theory(Ecs, Afun, ffun, P_, Q_, 3, L0, L1, np.zeros(4))
        if np.linalg.cond(M) > 1e5: skipped.append((seed, float(np.linalg.cond(M)))); continue
        cnt += 1
        Cb, Cr, cs = analytic_constants(Afun, P_, Q_, L0, L1, 3, M)
        eta = lambda t_: 1e-2 * np.array([np.sin(2 * t_ + i) + t_ * i for i in range(5)]); deta = lambda t_: 1e-2 * np.array([2 * np.cos(2 * t_ + i) + i for i in range(5)])
        beta = L0 @ eta(0.) + L1 @ eta(1.); rfun = lambda t_: Ecs @ deta(t_) - Afun(t_) @ eta(t_)
        lhs = sup_t(eta); rhs = Cb * np.linalg.norm(beta) + Cr * max(np.linalg.norm(rfun(t_)) for t_ in np.linspace(0, 1, 401))
        ok_lab &= bool(np.isfinite(rhs)) and lhs <= rhs
        print(f"      seed {seed}: |e|_inf={lhs:.2e} <= analytic bound={rhs:.2e}   (|A_red|<={cs['bred']:.2f}, gamma<={cs['gamma']:.2e}, a22<={cs['a22']:.3f})")
    print("      skipped (ill-conditioned M, cond>1e5):", skipped if skipped else "none")
    check(f"V9c Theorem 5 with the analytic inequalities of Lemma 2 evaluated in double precision (NumPy norms/exp: rigorous inequality, NOT interval-certified numbers) on {label} systems; ||M^+|| from a numerical solve", ok_lab and cnt == nsys)
# ---------------------------------------------------------------- V9d  the constant C_e is the same expression in the .tex source and in the script
import re, os, glob, hashlib
here = os.path.dirname(os.path.abspath(__file__))
tex_arg = [a for a in sys.argv[1:] if a.endswith('.tex')]
cands = sorted(glob.glob(os.path.join(here, 'proof_package*.tex')))
if tex_arg: texpath = tex_arg[0]
elif len(cands) == 1: texpath = cands[0]
else: texpath = None
if texpath is None or not os.path.exists(texpath):
    check("V9d source file: exactly one proof_package*.tex next to the script (or pass the path as an argument); refusing to guess", False, f"found {cands}")
    print("\nSUMMARY: source not found"); sys.exit(1)
tex = open(texpath).read()
print(f"   LaTeX source read: {texpath}  sha256={hashlib.sha256(tex.encode()).hexdigest()[:16]}")
missing = [m_ for m_ in ('lem:const', 'Rigorous versus sampled constants') if m_ not in tex]
if missing:
    check("V9d source file is the current one (contains the Lemma on rigorous constants)", False, f"missing markers {missing}: this .tex is an older file"); print("\nSUMMARY: outdated source"); sys.exit(1)
mm = re.search(r"C_e=\\nrm\{Q\}\\Big\[(.*?)\\Big\]", tex, re.S)
expr = mm.group(1)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
expr_py = re.sub(r'a_\{(\d\d)\}', r' a\1 ', expr.replace('\\gamma', ' gam '))
pe = parse_expr(expr_py, transformations=standard_transformations + (implicit_multiplication,))
gam, a12, a22, a21 = 1.7, 0.4, 0.6, 2.3
val_tex = float(pe.subs({'gam': gam, 'a12': a12, 'a22': a22, 'a21': a21})) * 1.9
val_script = 1.9 * (gam * (1 + a12) * (1 + a22 * a21) + a22)
disp = re.findall(r"Example (\d), \$C_\\beta\\le([\d.]+)\$, \$C_\\rho\\le([\d.]+)\$", tex)
ok_disp = len(disp) == 3 and all(float(cb) >= CERT[int(e_)][0] and float(cr) >= CERT[int(e_)][1] for e_, cb, cr in disp)
check("V9d displayed certified constants in the .tex are >= the interval upper endpoints (outward rounding)", ok_disp, str(disp))
check("V9d C_e = ||Q|| [ gamma(1+a12)(1+a22 a21) + a22 ] : expression parsed from proof_package.tex equals the one used in the script", abs(val_tex - val_script) < 1e-12, "tex: " + expr.strip())
# ---------------------------------------------------------------- V10  Corollary 1 (corrected): Examples 1,3,4 vs Example 2
print()
for ex in (1, 3, 4):
    A_ = np.array(Aconst[ex], float); Ar_ = A_[0, 0] - A_[0, 1] * A_[1, 0] / A_[1, 1]; col_ = np.array([1., -A_[1, 0] / A_[1, 1]])
    M_ = (col_ + np.array(L1d[ex], float) * col_ * np.exp(Ar_)).reshape(2, 1)
    check(f"V10 Example {ex}: k=2, r=1, rank M={np.linalg.matrix_rank(M_)} -> k - rank M = {2-np.linalg.matrix_rank(M_)} compatibility condition per column", np.linalg.matrix_rank(M_) == 1)
# Example 2: bounded solutions form a zero-parameter family; verified through the blow-up test (V6) and direct count
free_params = 1 - 1      # r = 1 differential variable x2, minus 1 condition x2(0) = -f2(0) imposed by boundedness of x1
check("V10 Example 2: bounded solutions have r - 1 = 0 free parameters, so BOTH boundary rows (k - 0 = 2) are compatibility conditions", free_params == 0 and 2 - free_params == 2)
# ---------------------------------------------------------------- V11  Example 2 endpoint formulas, independent verification
print()
tt_ = sp.symbols('tt')
def series_check(f1e, f2e, ordr=6):
    a_ = sp.cos(tt_) - sp.cos(3 * tt_) / sp.cos(tt_); b_ = f1e - f2e / sp.cos(tt_)
    c0_ = -f2e.subs(tt_, 0); us = sp.symbols('w1:%d' % (ordr + 1))
    x2_ = c0_ + sum(us[i] * tt_ ** (i + 1) for i in range(ordr))
    eq_ = sp.series(sp.diff(x2_, tt_) - a_ * x2_ - b_, tt_, 0, ordr).removeO()
    sol_ = sp.solve([sp.expand(eq_).coeff(tt_, i) for i in range(ordr)], us, dict=True)[0]
    x2_ = x2_.subs(sol_); g_ = -(sp.cos(3 * tt_) * x2_ + f2e); x1_ = sp.series(g_ / sp.sin(tt_), tt_, 0, 3).removeO()
    return sp.nsimplify(x2_.diff(tt_).subs(tt_, 0)), sp.nsimplify(x2_.diff(tt_, 2).subs(tt_, 0)), sp.nsimplify(x1_.coeff(tt_, 0)), sp.nsimplify(x1_.coeff(tt_, 1))
cases = {1: (sp.Integer(0), sp.exp(2 * tt_)), 2: (tt_ * sp.sin(tt_), sp.cos(2 * tt_))}
for kc, (f1e, f2e) in cases.items():
    x2p, x2pp, x10, x11 = series_check(f1e, f2e)
    p0, p1 = f1e.subs(tt_, 0), sp.diff(f1e, tt_).subs(tt_, 0); q0, q1, q2 = [sp.diff(f2e, tt_, i).subs(tt_, 0) for i in range(3)]
    th = (p0 - q0, p1 - q1, q0 - p0 - q1, -p1 / 2 - sp.Rational(9, 2) * q0 + q1 / 2 - q2 / 2)
    check(f"V11 Example 2 column {kc}: series solution gives x2'(0), x2''(0), x1(0), x1'(0) = {x2p}, {x2pp}, {x10}, {x11} ; theorem formulas give {th[0]}, {th[1]}, {th[2]}, {th[3]}",
          sp.simplify(x2p - th[0]) == 0 and sp.simplify(x2pp - th[1]) == 0 and sp.simplify(x10 - th[2]) == 0 and sp.simplify(x11 - th[3]) == 0)
    # numerical cross-check by high-precision ODE solve + difference quotients (independent of the series algebra)
    f1n = sp.lambdify(tt_, f1e, 'mpmath'); f2n = sp.lambdify(tt_, f2e, 'mpmath')
    solm = mp.odefun(lambda t_, y: (mp.cos(t_) - mp.cos(3 * t_) / mp.cos(t_)) * y + f1n(t_) - f2n(t_) / mp.cos(t_), 0, -f2n(mp.mpf(0)))
    x1m = lambda t_: -(mp.cos(3 * t_) * solm(t_) + f2n(t_)) / mp.sin(t_)
    h1 = mp.mpf('1e-7'); d_x1 = (x1m(2 * h1) - x1m(h1)) / h1 - 0   # slope between two small points ~ x1'(0)
    check(f"V11 Example 2 column {kc}: numerical slope of x1 near 0 = {float(d_x1):.5f} vs x1'(0) = {float(th[3]):.5f}", abs(d_x1 - float(th[3])) < 1e-3)
# sharpness of the regularity classes: f2 in C^1 but not C^2 (f2 = 1 + t^1.5) -> x1 continuous, x1' unbounded ~ t^(-1/2)
f2s_ = lambda t_: 1 + t_ ** 1.5
def rhs_sharp(t_, y): return [(np.cos(t_) - np.cos(3 * t_) / np.cos(t_)) * y[0] - f2s_(t_) / np.cos(t_)]
sol_s = solve_ivp(rhs_sharp, (0, 1), [-1.0], rtol=1e-13, atol=1e-15, dense_output=True)
x1s = lambda t_: -(np.cos(3 * t_) * sol_s.sol(t_)[0] + f2s_(t_)) / np.sin(t_)
slopes = []
for tv in (1e-2, 1e-3, 1e-4):
    h_ = tv / 2; slopes.append((x1s(tv + h_) - x1s(tv - h_)) / (2 * h_))
xs = np.array([1e-2, 1e-3, 1e-4]) ** -0.5                       # slope model  A + B t^(-1/2)
Bfit = (slopes[1] - slopes[0]) / (xs[1] - xs[0]); Bfit2 = (slopes[2] - slopes[1]) / (xs[2] - xs[1])
check("V11 sharpness: f2 in C^1 \\ C^2 => x1' ~ A + B t^(-1/2) with B = -1/2 (unbounded); two independent fits agree", abs(Bfit + 0.5) < 0.05 and abs(Bfit2 + 0.5) < 0.05, f"slopes {slopes[0]:.3f}, {slopes[1]:.3f}, {slopes[2]:.3f}; B fits {Bfit:.3f}, {Bfit2:.3f}")
from collections import Counter
print("\nSUMMARY: %d checks, %d passed, %d failed" % (len(RES), sum(RES), len(RES) - sum(RES)))
print("Kinds of check (not every check is an independent proof validation):")
for k_, c_ in sorted(Counter(KINDS).items(), key=lambda x: -x[1]): print(f"   {c_:3d}  {k_}")
sys.exit(0 if all(RES) else 1)
