"""Shared definition of the three-node coupled RLC descriptor benchmark (matrices, scenarios, references)."""
import numpy as np, scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.interpolate import BarycentricInterpolator

P = dict(C1=1.0e-3, C2=0.8e-3, C3=1.2e-3, L1=10e-3, L2=15e-3, Rl1=0.5, Rl2=0.8,
         Gs=0.10, G2=0.05, Gt=0.20, R1a=5.0, Rab=10.0, Rb3=5.0, Ra0=20.0, Rb0=25.0)
import os
T = 0.012*float(os.environ.get('KT','1'))   # optional horizon scaling
RAB = float(os.environ.get('RAB','1'))        # optional extra scaling of the bridge resistance Rab only

# sensitivity scaling (kR, kL, kC) from environment; default = nominal case.
# R-type parameters (winding, bridge, shunt resistances) scale by kR; the conductances Gs, G2, Gt scale by 1/kR.
KR, KL, KC = (float(v) for v in os.environ.get('SENS', '1,1,1').split(','))
for _k in ('Rl1', 'Rl2', 'R1a', 'Rab', 'Rb3', 'Ra0', 'Rb0'): P[_k] *= KR
for _k in ('Gs', 'G2', 'Gt'): P[_k] /= KR
# optional resistance-group scaling (env GRP = kW,kBr,kSh): W = winding resistances Rl1,Rl2; Br = coupling resistances R1a,Rab,Rb3;
# Sh = shunt-to-ground resistances Ra0,Rb0 and conductances Gs,G2,Gt (as 1/kSh)
KW, KBR, KSH = (float(v) for v in os.environ.get('GRP', '1,1,1').split(','))
for _k in ('Rl1', 'Rl2'): P[_k] *= KW
for _k in ('R1a', 'Rab', 'Rb3'): P[_k] *= KBR
for _k in ('Ra0', 'Rb0'): P[_k] *= KSH
for _k in ('Gs', 'G2', 'Gt'): P[_k] /= KSH
P['Rab'] *= RAB
for _k in ('L1', 'L2'): P[_k] *= KL
for _k in ('C1', 'C2', 'C3'): P[_k] *= KC
NAMES = ['v1', 'v2', 'v3', 'i1', 'i2', 'va', 'vb']

def hand_matrices(p=P):
    E = np.diag([p['C1'], p['C2'], p['C3'], p['L1'], p['L2'], 0.0, 0.0])
    A = np.zeros((7, 7))
    A[0, 0] = -p['Gs'] - 1/p['R1a']; A[0, 3] = -1; A[0, 5] = 1/p['R1a']
    A[1, 1] = -p['G2']; A[1, 3] = 1; A[1, 4] = -1
    A[2, 2] = -p['Gt'] - 1/p['Rb3']; A[2, 4] = 1; A[2, 6] = 1/p['Rb3']
    A[3, 0] = 1; A[3, 1] = -1; A[3, 3] = -p['Rl1']
    A[4, 1] = 1; A[4, 2] = -1; A[4, 4] = -p['Rl2']
    A[5, 0] = 1/p['R1a']; A[5, 5] = -(1/p['R1a'] + 1/p['Ra0'] + 1/p['Rab']); A[5, 6] = 1/p['Rab']
    A[6, 2] = 1/p['Rb3']; A[6, 5] = 1/p['Rab']; A[6, 6] = -(1/p['Rab'] + 1/p['Rb0'] + 1/p['Rb3'])
    B = np.zeros((7, 2)); B[0, 0] = 1; B[2, 1] = 1
    return E, A, B

E, A, B = hand_matrices()
A11, A12, A21, A22 = A[:5, :5], A[:5, 5:], A[5:, :5], A[5:, 5:]
Ed = E[:5, :5]; Bd = B[:5]
Ared = A11 - A12 @ np.linalg.solve(A22, A21)
Mred = np.linalg.solve(Ed, Ared)
Phi = sla.expm(Mred * T)
M_REC = -np.linalg.solve(A22, A21)           # x_a = M_REC x_d  (exact algebraic recovery)

m = 3
PH = np.array([0.0, 0.8, 1.6]); AMP = np.array([1.0, 0.8, 1.2])
def u_of(t):
    t = np.atleast_1d(t)[:, None]
    u1 = AMP*(0.80 + 0.30*np.sin(2*np.pi*t/T + PH)); u3 = 0.10 + 0.05*np.cos(2*np.pi*t/T + PH)
    return np.stack([u1, u3], 0)               # (2, len(t), 3)
V0 = np.array([[3.0, 2.8, 3.2], [2.6, 2.4, 2.8], [2.2, 2.0, 2.4]])
IT = np.array([[0.50, 0.45, 0.55], [0.38, 0.34, 0.42]])

# characteristic scales from the DC operating point of the mean sources (u1=0.8, u3=0.1)
xss = -np.linalg.solve(Ared, Bd @ np.array([0.8, 0.1]))
VSC = float(np.abs(xss[:3]).max()); ISC = float(np.abs(xss[3:]).max())

def _f(t, x, j): return Mred @ x + np.linalg.solve(Ed, Bd @ u_of(t)[:, 0, j])
def superposition(j):
    opt = dict(rtol=1e-12, atol=1e-14, method='DOP853', dense_output=True)
    xpT = solve_ivp(lambda t, x: _f(t, x, j), (0, T), np.zeros(5), **opt).sol(T)
    z = np.linalg.solve(Phi[3:5, 3:5], IT[:, j] - xpT[3:5] - Phi[3:5, :3] @ V0[:, j])
    return solve_ivp(lambda t, x: _f(t, x, j), (0, T), np.r_[V0[:, j], z], **opt)
_SOL = [superposition(j) for j in range(m)]
def reference(tq):
    """Full 7-state reference on grid tq: shape (7, len(tq), 3)"""
    out = np.zeros((7, len(tq), m))
    for j in range(m):
        xd = _SOL[j].sol(tq); out[:5, :, j] = xd; out[5:, :, j] = M_REC @ xd
    return out

def cheb(N):
    x = np.cos(np.pi*np.arange(N+1)/N); c = np.r_[2, np.ones(N-1), 2]*(-1)**np.arange(N+1)
    X = np.tile(x, (N+1, 1)).T; dX = X - X.T
    D = np.outer(c, 1/c)/(dX + np.eye(N+1)); return D - np.diag(D.sum(1)), x
def cheb_full_descriptor(j, N):
    """Chebyshev collocation on the FULL 7x7 descriptor system (no elimination)."""
    D, x = cheb(N); t = T*(1 - x)/2; Dt = -(2/T)*D
    M = np.kron(E, Dt) - np.kron(A, np.eye(N+1)); rhs = (B @ u_of(t)[:, :, j]).reshape(-1)
    for r in range(3): i = r*(N+1); M[i] = 0; M[i, i] = 1; rhs[i] = V0[r, j]
    for r in (3, 4): i = r*(N+1) + N; M[i] = 0; M[i, i] = 1; rhs[i] = IT[r-3, j]
    X = np.linalg.solve(M, rhs).reshape(7, N+1)
    return t, X, Dt
