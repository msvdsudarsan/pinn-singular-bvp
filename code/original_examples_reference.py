import numpy as np
from scipy.integrate import solve_ivp
np.set_printoptions(precision=5,suppress=True)
def F(ex,t):
    return {1:np.array([[t,np.exp(t)],[np.cos(t),t**2]]),
            2:np.array([[0,t*np.sin(t)],[np.exp(2*t),np.cos(2*t)]]),
            3:np.array([[np.sin(t),np.exp(t)],[0,0]]),
            4:np.array([[np.cos(t),t],[0,0]])}[ex]
A={1:np.array([[1,2],[3,4.]]),3:np.array([[0,1],[1,1.]]),4:np.array([[1,1],[2,2.]])}
L1={1:np.diag([-2,1.]),2:np.diag([-2,1.]),3:np.diag([-1,1.]),4:np.diag([-1,1.])}
W={1:np.array([[0.6887,-0.9577],[4.5984,7.6409]]),2:np.array([[39.6326,7.6130],[-10.8047,-4.2308]]),
   3:np.array([[-0.6506,-0.8591],[0.3494,-1.8591]]),4:np.array([[-0.8415,-0.5],[-0.8415,-0.5]])}
def full(ex,k,a):
    """returns function t->(x1,x2) for column k, differential initial value a"""
    Am=A[ex]
    def rhs(t,y):
        f=F(ex,t)[:,k]; x2=-(Am[1,0]*y[0]+f[1])/Am[1,1]
        return [Am[0,0]*y[0]+Am[0,1]*x2+f[0]]
    s=solve_ivp(rhs,(0,1),[a],rtol=1e-12,atol=1e-14,dense_output=True)
    def X(t):
        x1=s.sol(t)[0];f=F(ex,t)[:,k];return np.array([x1,-(Am[1,0]*x1+f[1])/Am[1,1]])
    return X
print("== Examples 1,3,4: scalar reduction x1'=..., x2 algebraic; BC has 2 rows, 1 free parameter a")
for ex in (1,3,4):
    sol=[]
    for k in range(2):
        X0=full(ex,k,0.0);X1=full(ex,k,1.0)
        # BC row r: X(0)[r]+L1[r,r]*X(1)[r] = W[r,k]; linear in a
        b0=np.array([X0(0)[r]+L1[ex][r,r]*X0(1)[r] for r in range(2)])
        b1=np.array([X1(0)[r]+L1[ex][r,r]*X1(1)[r] for r in range(2)])-b0
        rhs=W[ex][:,k]-b0
        a=np.linalg.lstsq(b1.reshape(2,1),rhs,rcond=None)[0][0]
        res=rhs-b1*a
        print(f"Ex{ex} col{k+1}: a_LS={a:.6f}  BC-row sensitivity d(row)/da={b1}  BC residuals={res}")
        sol.append(a)
# ---- Example 2 (removable singularity at t=0) ----
import numpy as np
from scipy.integrate import solve_ivp


np.set_printoptions(precision=5,suppress=True)
W=np.array([[39.6326,7.6130],[-10.8047,-4.2308]])
F=lambda t:np.array([[0,t*np.sin(t)],[np.exp(2*t),np.cos(2*t)]])
dF=lambda t:np.array([[0,np.sin(t)+t*np.cos(t)],[2*np.exp(2*t),-2*np.sin(2*t)]])
def col(k,c):
    def rhs(t,y):
        f=F(t)[:,k]; return [(np.cos(t)-np.cos(3*t)/np.cos(t))*y[0]-f[1]/np.cos(t)+f[0]]
    s=solve_ivp(rhs,(0,1),[c],rtol=1e-12,atol=1e-14,dense_output=True)
    return s,rhs
out=[]
for k in range(2):
    c=-F(0)[1,k]                       # forced by boundedness of x1 at t=0
    s,rhs=col(k,c)
    x2=lambda t:s.sol(t)[0]
    x1=lambda t:-(np.cos(3*t)*x2(t)+F(t)[1,k])/np.sin(t)
    x1_0=-(rhs(0,[c])[0]+dF(0)[1,k])          # L'Hopital limit
    x1_1=x1(1.0);x2_0=c;x2_1=x2(1.0)
    r1=x1_0-2*x1_1; r2=x2_0+x2_1
    print(f"col{k+1}: x2(0)=-F2(0)={c:.4f}, x1(0)={x1_0:.4f}, x1(1)={x1_1:.4f}, x2(1)={x2_1:.4f} -> BC rows = [{r1:.4f},{r2:.4f}]  given W = {W[:,k]}")
    # unbounded case: c != -F2(0): x1 ~ -(c+F2(0))/t  near 0
    for cc in (c+0.1,):
        s2,_=col(k,cc); t=1e-3
        print("   if x2(0)=%.3f instead: x1(1e-3)=%.1f (blows up like 1/t)"%(cc,-(np.cos(3*t)*s2.sol(t)[0]+F(t)[1,k])/np.sin(t)))
