"""Extension A (mechanism check): conditioning of the discretised least-squares operator for the full (D) and reduced (R) residual.
For each Rab scale: cond of the scaled collocation operator acting on the free unknowns (hard boundary conditions eliminated)."""
import sys, os, numpy as np, json
import network_core as nc
S=None
def conds(N=40):
    E,A,B=nc.E,nc.A,nc.B; T=nc.T
    VS,IS=nc.VSC,nc.ISC
    Sc=np.array([VS]*3+[IS]*2+[VS]*2); RS=np.array([1/IS]*3+[1/VS]*2+[1/IS]*2)
    D,x=nc.cheb(N); Dt=-(2/T)*D
    Ms=np.kron(np.diag(RS)@E@np.diag(Sc),Dt)-np.kron(np.diag(RS)@A@np.diag(Sc),np.eye(N+1))
    bc=[r*(N+1) for r in range(3)]+[r*(N+1)+N for r in (3,4)]
    free=[i for i in range(7*(N+1)) if i not in bc]
    cD=np.linalg.cond(Ms[:,free])
    # reduced: x_a = Mrec x_d  -> differential rows only
    Mrec=nc.M_REC
    Ed=nc.Ed; Ar=nc.Ared; Rd=np.diag(RS[:5]); Sd=np.diag(Sc[:5])
    Mr=np.kron(Rd@Ed@Sd,Dt)-np.kron(Rd@Ar@Sd,np.eye(N+1))
    bcr=[r*(N+1) for r in range(3)]+[r*(N+1)+N for r in (3,4)]
    freer=[i for i in range(5*(N+1)) if i not in bcr]
    cR=np.linalg.cond(Mr[:,freer])
    return cD,cR,np.linalg.cond(nc.A22)
if __name__=="__main__":
    print(json.dumps(dict(rab=nc.RAB,cond_D=conds()[0],cond_R=conds()[1],cond_A22=conds()[2])))
