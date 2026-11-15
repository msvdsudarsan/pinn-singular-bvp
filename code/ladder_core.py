"""Extension C: n-node RLC ladder descriptor benchmark with n_a = n-1 algebraic (capacitor-free) nodes.
States x = [v_1..v_n, i_1..i_{n-1}, a_1..a_{n-1}].  Capacitive node k: C_k v_k' = -Gs_k v_k + i_{k-1} - i_k + (a_k - v_k)/Rc [k<=n-1] + u_k (u only at nodes 1 and n)
Inductor k: L_k i_k' = v_k - v_{k+1} - Rl_k i_k.  Algebraic node a_k: 0 = (v_k - a_k)/Rc - a_k/R0 + (a_{k+1}-a_k)/Rab [k<n-1] + (a_{k-1}-a_k)/Rab [k>1].
Boundary data: voltages at t=0, inductor currents at t=T.  Reference: exact solution of the reduced linear ODE (matrix exponential / superposition),
independent check: Chebyshev collocation of the FULL descriptor system."""
import os, numpy as np, scipy.linalg as sla
from scipy.integrate import solve_ivp
NN = int(os.environ.get('NNODE', '6'))
T = 0.012
RAB = float(os.environ.get('RAB', '1'))
def build(n=NN):
    na = n-1; N = n + (n-1) + na
    C = 1e-3*(1 + 0.15*np.sin(np.arange(n)*1.3)); L = 10e-3*(1 + 0.2*np.cos(np.arange(n-1)*0.9)); Rl = 0.5*(1 + 0.1*np.arange(n-1) % 0.4)
    Gs = 0.10*np.ones(n); Rc = 5.0; R0 = 20.0; Rab = 10.0*RAB
    E = np.zeros((N, N)); A = np.zeros((N, N)); Bm = np.zeros((N, 2))
    iv = lambda k: k; ii = lambda k: n+k; ia = lambda k: n+(n-1)+k
    for k in range(n):
        E[iv(k), iv(k)] = C[k]; A[iv(k), iv(k)] -= Gs[k]
        if k > 0: A[iv(k), ii(k-1)] += 1
        if k < n-1: A[iv(k), ii(k)] -= 1; A[iv(k), iv(k)] -= 1/Rc; A[iv(k), ia(k)] += 1/Rc
    Bm[iv(0), 0] = 1; Bm[iv(n-1), 1] = 1
    for k in range(n-1):
        E[ii(k), ii(k)] = L[k]; A[ii(k), iv(k)] += 1; A[ii(k), iv(k+1)] -= 1; A[ii(k), ii(k)] -= Rl[k]
        A[ia(k), iv(k)] += 1/Rc; A[ia(k), ia(k)] -= 1/Rc + 1/R0
        if k < na-1: A[ia(k), ia(k+1)] += 1/Rab; A[ia(k), ia(k)] -= 1/Rab
        if k > 0: A[ia(k), ia(k-1)] += 1/Rab; A[ia(k), ia(k)] -= 1/Rab
    return E, A, Bm, n, na
E, A, B, n, na = build()
nd = n + (n-1); N = nd + na
Ed, A11, A12, A21, A22 = E[:nd, :nd], A[:nd, :nd], A[:nd, nd:], A[nd:, :nd], A[nd:, nd:]
Ared = A11 - A12 @ np.linalg.solve(A22, A21); Mred = np.linalg.solve(Ed, Ared); Bd = B[:nd]
M_REC = -np.linalg.solve(A22, A21)
def u_of(t):
    t = np.atleast_1d(t); return np.stack([0.8 + 0.3*np.sin(2*np.pi*t/T), 0.1 + 0.05*np.cos(2*np.pi*t/T)], 0)
V0 = 2.0 + 0.2*np.sin(np.arange(n)); IT = 0.4 + 0.05*np.cos(np.arange(n-1))
xss = -np.linalg.solve(Ared, Bd @ np.array([0.8, 0.1])); VSC = float(np.abs(xss[:n]).max()); ISC = float(np.abs(xss[n:]).max())
Phi = sla.expm(Mred*T)
def _f(t, x): return Mred @ x + np.linalg.solve(Ed, Bd @ u_of(t)[:, 0])
opt = dict(rtol=1e-12, atol=1e-14, method='DOP853', dense_output=True)
xpT = solve_ivp(_f, (0, T), np.zeros(nd), **opt).sol(T)
z = np.linalg.solve(Phi[n:nd, n:nd], IT - xpT[n:nd] - Phi[n:nd, :n] @ V0)
_SOL = solve_ivp(_f, (0, T), np.r_[V0, z], **opt)
def reference(tq):
    xd = _SOL.sol(tq); return np.vstack([xd, M_REC @ xd])
def cheb(Nc):
    x = np.cos(np.pi*np.arange(Nc+1)/Nc); c = np.r_[2, np.ones(Nc-1), 2]*(-1)**np.arange(Nc+1)
    X = np.tile(x, (Nc+1, 1)).T; dX = X - X.T; D = np.outer(c, 1/c)/(dX + np.eye(Nc+1)); return D - np.diag(D.sum(1)), x
def cheb_full(Nc=60):
    D, x = cheb(Nc); t = T*(1-x)/2; Dt = -(2/T)*D
    M = np.kron(E, Dt) - np.kron(A, np.eye(Nc+1)); rhs = (B @ u_of(t)).reshape(-1)
    for r in range(n): i = r*(Nc+1); M[i] = 0; M[i, i] = 1; rhs[i] = V0[r]
    for r in range(n, nd): i = r*(Nc+1)+Nc; M[i] = 0; M[i, i] = 1; rhs[i] = IT[r-n]
    return t, np.linalg.solve(M, rhs).reshape(N, Nc+1)
if __name__ == "__main__":
    t, X = cheb_full(60); ref = reference(t); idx = np.argsort(t)
    print('n', n, 'states', N, 'index-1 (A22 invertible) cond(A22)=%.2f' % np.linalg.cond(A22), 'rank(E)=', np.linalg.matrix_rank(E),
          'rel diff full-descriptor Chebyshev vs reduced reference = %.2e' % (np.linalg.norm(X-ref)/np.linalg.norm(ref)), 'cond(Phi_block)=%.2e' % np.linalg.cond(Phi[n:nd, n:nd]))
