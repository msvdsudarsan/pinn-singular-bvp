"""Validity checks for one (kR,kL,kC) case: positivity, rank E, A22, index-1, BC operator, reference solver vs full-descriptor collocation."""
import json, numpy as np, scipy.linalg as sla
import network_core as nc
p = nc.P
pos = all(v > 0 for v in p.values())
rankE = int(np.linalg.matrix_rank(nc.E))
A22 = nc.A22; condA22 = float(np.linalg.cond(A22)); detA22 = float(np.linalg.det(A22))
# BC operator on the 5 reduced states: x1..3(0) and i1,i2(T); in terms of Phi: M = [[I3,0],[Phi[3:5,:3],Phi[3:5,3:5]]]
Mbc = np.block([[np.eye(3), np.zeros((3, 2))], [nc.Phi[3:5, :3], nc.Phi[3:5, 3:5]]])
condM = float(np.linalg.cond(Mbc)); rankM = int(np.linalg.matrix_rank(Mbc))
eig = np.linalg.eigvals(nc.Mred); 
tq = np.linspace(0, nc.T, 501); ref = nc.reference(tq)
errs = []
for j in range(nc.m):
    t, X, _ = nc.cheb_full_descriptor(j, 60)
    from scipy.interpolate import BarycentricInterpolator
    Xi = np.stack([BarycentricInterpolator(t[::-1], X[k][::-1])(tq) for k in range(7)])
    errs.append(float(np.abs(Xi - ref[:, :, j]).max()/np.abs(ref[:, :, j]).max()))
print(json.dumps(dict(RAB=nc.RAB, T=nc.T, KW=nc.KW, KBR=nc.KBR, KSH=nc.KSH, KR=nc.KR, KL=nc.KL, KC=nc.KC, positive=pos, rankE=rankE, condA22=condA22, detA22=detA22,
    rankM=rankM, condM=condM, max_re_eig=float(eig.real.max()), min_re_eig=float(eig.real.min()),
    ref_vs_cheb_maxrel=max(errs), VSC=nc.VSC, ISC=nc.ISC)))
