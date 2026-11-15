import sys,time,json,numpy as np
import jax,jax.numpy as jnp
jax.config.update("jax_enable_x64",True)
from jax.flatten_util import ravel_pytree
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
# ---------- problem data ----------
Em={1:[[1,0],[0,0]],2:[[0,1],[0,0]],3:[[1,0],[0,0]],4:[[1,0],[0,0]]}
Wd={1:[[0.6887,-0.9577],[4.5984,7.6409]],2:[[39.6326,7.6130],[-10.8047,-4.2308]],3:[[-0.6506,-0.8591],[0.3494,-1.8591]],4:[[-0.8415,-0.5],[-0.8415,-0.5]]}
L1d={1:[-2,1],2:[-2,1],3:[-1,1],4:[-1,1]}
Ac={1:[[1,2],[3,4]],3:[[0,1],[1,1]],4:[[1,1],[2,2]]}
def A_t(ex,t):
    if ex==2: return jnp.array([[jnp.tan(t),jnp.cos(t)],[jnp.sin(t),jnp.cos(3*t)]])
    return jnp.array(Ac[ex],dtype=jnp.float64)
def F_t(ex,t):
    z=jnp.zeros_like(t)
    if ex==1: return jnp.array([[t,jnp.exp(t)],[jnp.cos(t),t**2]])
    if ex==2: return jnp.array([[z,t*jnp.sin(t)],[jnp.exp(2*t),jnp.cos(2*t)]])
    if ex==3: return jnp.array([[jnp.sin(t),jnp.exp(t)],[z,z]])
    return jnp.array([[jnp.cos(t),t],[z,z]])
# ---------- independent reference (reduction to scalar ODE) ----------
def np_F(ex,t):
    z=0*t
    return {1:np.array([[t,np.exp(t)],[np.cos(t),t**2]]),2:np.array([[z,t*np.sin(t)],[np.exp(2*t),np.cos(2*t)]]),
            3:np.array([[np.sin(t),np.exp(t)],[z,z]]),4:np.array([[np.cos(t),t],[z,z]])}[ex]
tt=np.linspace(0,1,501)
def reference(ex):
    R=np.zeros((501,2,2))
    L1=np.diag(L1d[ex]);W=np.array(Wd[ex],float)
    for k in range(2):
        if ex!=2:
            Am=np.array(Ac[ex],float)
            def mk(a):
                def rhs(t,y):
                    f=np_F(ex,t)[:,k];x2=-(Am[1,0]*y[0]+f[1])/Am[1,1];return [Am[0,0]*y[0]+Am[0,1]*x2+f[0]]
                s=solve_ivp(rhs,(0,1),[a],rtol=1e-12,atol=1e-14,dense_output=True)
                def X(t):
                    x1=s.sol(t)[0];f=np_F(ex,t)[:,k];return np.array([x1,-(Am[1,0]*x1+f[1])/Am[1,1]])
                return X
            X0,X1=mk(0.),mk(1.)
            b0=np.array([X0(0.)[r]+L1[r,r]*X0(1.)[r] for r in range(2)])
            b1=np.array([X1(0.)[r]+L1[r,r]*X1(1.)[r] for r in range(2)])-b0
            a=np.linalg.lstsq(b1.reshape(2,1),W[:,k]-b0,rcond=None)[0][0]
            Xf=mk(a)
            for n,t in enumerate(tt): R[n,:,k]=Xf(t)
        else:
            F=lambda t:np_F(2,t)[:,k]
            c=-np_F(2,0.)[1,k]
            def rhs(t,y): f=F(t);return [(np.cos(t)-np.cos(3*t)/np.cos(t))*y[0]-f[1]/np.cos(t)+f[0]]
            s=solve_ivp(rhs,(0,1),[c],rtol=1e-12,atol=1e-14,dense_output=True)
            dF2=np.array([2.0,-0.0])[k]  # d/dt F2 at 0: 2e^{0}=2 (col0), -2 sin0=0 (col1)
            for n,t in enumerate(tt):
                y=s.sol(t)[0]
                x1=-(rhs(0.,[c])[0]+dF2) if n==0 else -(np.cos(3*t)*y+F(t)[1])/np.sin(t)
                R[n,:,k]=[x1,y]
    return R
REF={ex:reference(ex) for ex in (1,2,3,4)}
# ---------- PINN ----------
def init(key,w=64,d=4):
    sizes=[1]+[w]*d+[4];ps=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        key,k=jax.random.split(key);ps.append((jax.random.normal(k,(b,a))*np.sqrt(2/(a+b)),jnp.zeros(b)))
    return ps
def rawnet(ps,t):
    h=jnp.atleast_1d(t)
    for W,b in ps[:-1]: h=jnp.tanh(W@h+b)
    W,b=ps[-1];return (W@h+b).reshape(2,2)
def Xof(ps,t,ex,hard):
    N=rawnet(ps,t)
    if hard:   # X(t)=N(t)+(1-t)(W-N(0)-L1 N(1)): exact L0X(0)+L1X(1)=W (L0=I) for ANY network N
        L1=jnp.diag(jnp.array(L1d[ex],dtype=jnp.float64))
        return N+(1-t)*(jnp.array(Wd[ex],dtype=jnp.float64)-rawnet(ps,0.0)-L1@rawnet(ps,1.0))
    return N
def resid(ps,t,ex,hard):
    X,dX=jax.jvp(lambda s:Xof(ps,s,ex,hard),(t,),(1.0,))
    return jnp.array(Em[ex],dtype=jnp.float64)@dX-A_t(ex,t)@X-F_t(ex,t)
tc=jnp.linspace(0,1,101)
CACHE={}
def build(ex,hard):
    if (ex,hard) in CACHE: return CACHE[(ex,hard)]
    ps0=init(jax.random.PRNGKey(0));flat0,unr=ravel_pytree(ps0)
    L1=jnp.diag(jnp.array(L1d[ex],dtype=jnp.float64));Wm=jnp.array(Wd[ex],dtype=jnp.float64)
    def loss(f):
        ps=unr(f);r=jax.vmap(lambda t:resid(ps,t,ex,hard))(tc)
        lo=jnp.mean(r**2)
        bc=Xof(ps,0.0,ex,hard)+L1@Xof(ps,1.0,ex,hard)-Wm
        return lo+100.0*jnp.mean(bc**2)
    lg=jax.jit(jax.value_and_grad(loss))
    @jax.jit
    def step(f,m,v,k):
        l,g=jax.value_and_grad(loss)(f)
        m=.9*m+.1*g;v=.999*v+.001*g**2
        return f-1e-3*(m/(1-.9**k))/(jnp.sqrt(v/(1-.999**k))+1e-8),m,v,l
    @jax.jit
    def evalfun(f):
        ps=unr(f)
        X=jax.vmap(lambda t:Xof(ps,t,ex,hard))(jnp.asarray(tt))
        r=jax.vmap(lambda t:resid(ps,t,ex,hard))(jnp.asarray(tt))
        bc=Xof(ps,0.0,ex,hard)+L1@Xof(ps,1.0,ex,hard)-Wm
        return X,r,bc
    CACHE[(ex,hard)]=(unr,lg,step,evalfun);return CACHE[(ex,hard)]
def run(ex,seed,mode,hard,N=2000,frac=0.3):
    unr,lg,step,evalfun=build(ex,hard)
    flat,_=ravel_pytree(init(jax.random.PRNGKey(1000*ex+seed)))
    na={'adam':N,'lbfgs':0,'two':int(round(frac*N))}[mode];nl=N-na
    t0=time.time();m=jnp.zeros_like(flat);v=jnp.zeros_like(flat)
    for k in range(1,na+1): flat,m,v,l=step(flat,m,v,float(k))
    nfev=0
    if nl>0:
        fun=lambda x:tuple(np.asarray(a,dtype=np.float64) for a in lg(jnp.asarray(x)))
        r=minimize(fun,np.asarray(flat),jac=True,method='L-BFGS-B',options=dict(maxfun=nl,maxiter=nl,maxcor=50,ftol=1e-18,gtol=1e-14))
        flat=jnp.asarray(r.x);nfev=int(r.nfev)
    sec=time.time()-t0
    X,r,bc=evalfun(flat);X=np.asarray(X);r=np.asarray(r);bc=np.asarray(bc)
    ref=REF[ex];e=X-ref
    rel=lambda a,b:float(np.linalg.norm(a)/np.linalg.norm(b))
    out=dict(ex=ex,seed=seed,mode=mode,hard=hard,frac=frac if mode=='two' else (0.0 if mode=='lbfgs' else 1.0),
        relL2_total=rel(e,ref),relL2_x1=rel(e[:,0,:],ref[:,0,:]),relL2_x2=rel(e[:,1,:],ref[:,1,:]),max_abs=float(np.abs(e).max()),
        bc_max=float(np.abs(bc).max()),diff_res_rms=float(np.sqrt(np.mean(r[:,0,:]**2))),alg_res_rms=float(np.sqrt(np.mean(r[:,1,:]**2))),
        final_loss=float(lg(flat)[0]),adam_evals=na,lbfgs_nfev=nfev,sec=sec)
    out['fail']=bool(out['relL2_total']>5e-2 or not np.isfinite(out['relL2_total']))
    return out
if __name__=="__main__":
    s0,s1=int(sys.argv[1]),int(sys.argv[2])
    cfg=[('adam',False),('lbfgs',False),('two',False),('adam',True),('lbfgs',True),('two',True)]
    for sd in range(s0,s1):
        for ex in (1,2,3,4):
            for mode,hard in cfg:
                print(json.dumps(run(ex,sd,mode,hard)),flush=True)
