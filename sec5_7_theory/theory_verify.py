# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""Numerical closed loop for the falsifiable theoretical predictions (reuse trained weights, no retraining):
Exp1 error floor: fix K=10, zero-shot super-resolution 17->65->129->257->513; the error should saturate near 129 (approach a constant floor).
Exp2 s-scan: fix alpha=.85; the ratio err_FNO/err_FrFNO should grow monotonically with s (~K^{2s}, K=10).
Unique-eigenvalue propagator (mathematically equivalent to 2-D mode-by-mode propagation, memory O(N) instead of O(N^2)*Nt); incremental saving allows resume."""
import os, sys, time
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
import numpy as np, torch
import frfno_core as FC
from frfno_core import DualNet, DEV, DT, prop_at
from gpu_solver import spectral_eigvals_torch, dst2_torch
from solver_spacetime_fractional import _l1_b, spectral_eigvals
from nl_solver import gpu_solve_nl_batch
import surrogate_compare as SC
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
import d_scan as DS

TN, BETA = 0.015, 3.0; FC.T = TN; SC.T = TN
NTE = 24
OUT_TXT = os.path.join(ROOT, 'theory_verify_result.txt')
OUT_NPZ = os.path.join(ROOT, 'theory_verify.npz')
L = []
def log(s): print(s, flush=True); L.append(str(s)); open(OUT_TXT,'w',encoding='utf-8').write('\n'.join(L))
def dump(**kw):
    d=dict(np.load(OUT_NPZ)) if os.path.exists(OUT_NPZ) else {}
    d.update({k:np.asarray(v) for k,v in kw.items()}); np.savez(OUT_NPZ,**d)

@torch.no_grad()
def build_pt_eig(Nx, Nt, AD, SD, M=1600):
    """Place a uniform 1-D grid over the smooth argument lam=q^s for the L1 recursion (g is smooth in lam, avoiding the
    s<1 algebraic singularity at lambda=0), then interpolate back to 2-D. Memory Nt*ns*M, independent of resolution N.
    Returns (AD,SD,PT[na,ns,n,n]) compatible with prop_at."""
    h=1.0/Nx; m=np.arange(1,Nx); qx=(4.0/h**2)*np.sin(np.pi*h*m/2.0)**2
    q2=(qx[:,None]+qx[None,:]); n=Nx-1; SD=np.asarray(SD); ns=len(SD)
    base=np.linspace(0.0,1.0,M)[None]                       # common [0,1] grid
    lam_max=(q2.max()**SD)[:,None]                          # per-s upper bound of lam=q^s
    lam_grid=base*lam_max                                   # (ns,M); each row is uniform in lam
    lamf=torch.tensor(lam_grid,device=DEV,dtype=DT)
    PT=np.zeros((len(AD),ns,n,n),np.float32)
    for ia,al in enumerate(AD):
        b=torch.tensor(_l1_b(al,Nt),device=DEV,dtype=DT)
        dta=(TN/Nt)**al; hist=torch.empty((Nt,ns,M),device=DEV,dtype=DT); hist[0]=1.0; v=hist[0]
        for k in range(1,Nt+1):
            rhs=b[k-1]*hist[0]
            if k>1:
                jj=torch.arange(1,k,device=DEV); rhs=rhs+torch.einsum('j,jxy->xy',b[k-jj-1]-b[k-jj],hist[1:k])
            v=rhs/(b[0]+dta*lamf)
            if k<Nt: hist[k]=v
        vn=v.cpu().numpy(); del hist
        tq=q2.ravel()
        for js in range(ns):
            PT[ia,js]=np.interp(tq**SD[js], lam_grid[js], vn[js]).reshape(n,n)
        torch.cuda.empty_cache()
    return AD,SD,PT

def true_field_batched(Nx,Nt,seeds,a,s,xi,chunk=6):
    if (Nx,'Phi') not in PHI_CACHE:
        _p,_e=kl_basis(Nx,KMAX); PHI_CACHE[(Nx,'Phi')],PHI_CACHE[(Nx,'e')]=_p,_e
    Phi,e=PHI_CACHE[(Nx,'Phi')],PHI_CACHE[(Nx,'e')]
    U0=np.stack([DS.generate_multiscale_initial(Nx,8,int(sd)) for sd in seeds]).astype(np.float32)
    NU=make_nu(Nx,xi,Phi,e); Q=spectral_eigvals_torch(Nx,DEV,DT); Fs=[]
    for i0 in range(0,len(a),chunk):
        with torch.no_grad():
            f=gpu_solve_nl_batch(U0[i0:i0+chunk],NU[i0:i0+chunk],a[i0:i0+chunk],s[i0:i0+chunk],
                                 TN,Nt,Nx,beta=BETA,Q=Q,device=DEV,dtype=DT).cpu().numpy()
        Fs.append(f.reshape(len(a[i0:i0+chunk]),-1))
    return np.concatenate(Fs,0),U0,NU

def load_pair(wpath, tagA=1, tagF=2):
    W=torch.load(wpath,map_location=DEV)
    a=DualNet(7,seed=tagA).to(DEV).float(); a.load_state_dict(W['FrFNO']); a.eval()
    f=DualNet(2,field_dim=0,seed=tagF).to(DEV).float(); f.load_state_dict(W['FNO']); f.eval()
    return a,f

def err_of(net,kind,Nx,U0,NU,a,s,Ftr,PT):
    em,_,_=SC.eval_spectral(net,kind,Nx,U0,NU,a,s,Ftr,PT); return em

def exp1(done):
    log('\n########## Exp1 error floor (fixed K=10, B1 weights, matched dt refinement) ##########')
    fr,fno=load_pair(os.path.join(ROOT, 'b1_weights.pt'))
    SC.MEAN, SC.SV = 0.0051, 0.2266   # B1 training ground-truth statistics (see b1_full.log)
    RES=[(16,400),(64,800),(128,1600),(256,3200)]
    rng=np.random.RandomState(2024); seeds=rng.randint(100000,size=NTE)
    a=rng.uniform(0.55,0.95,NTE).astype(np.float32); s=rng.uniform(0.35,0.78,NTE).astype(np.float32)
    xi=rng.randn(NTE,KMAX).astype(np.float32)
    AD=np.linspace(0.45,1.05,21); SD=np.linspace(0.25,0.90,21)
    res=[]; eA=[]; eF=[]
    for Nx,Nt in RES:
        if Nx in done: continue
        t0=time.time(); PT=build_pt_eig(Nx,Nt,AD,SD); tpt=time.time()-t0
        t0=time.time(); Ftr,U0,NU=true_field_batched(Nx,Nt,seeds,a,s,xi,chunk=2 if Nx>=256 else NTE); ttf=time.time()-t0
        t0=time.time()
        with torch.no_grad():
            ea=err_of(fr,'A',Nx,U0,NU,a,s,Ftr,PT); ef=err_of(fno,'F',Nx,U0,NU,a,s,Ftr,PT)
        tev=time.time()-t0
        log('  N=%3d^2  FrFNO=%6.3f%%  FNO=%6.3f%%  ratio=%5.2f  (table %.0fs/ground-truth %.0fs/eval %.0fs)'
            %(Nx+1,ea,ef,ef/ea,tpt,ttf,tev))
        res.append(Nx+1); eA.append(ea); eF.append(ef)
        dump(exp1_res=np.array(res),exp1_fr=np.array(eA),exp1_fno=np.array(eF))
    return np.array(res),np.array(eA),np.array(eF)

def exp2(done):
    log('\n########## Exp2 s-scan (B4 weights, alpha=.85, 129^2, advantage ~K^{2s}) ##########')
    fr,fno=load_pair(os.path.join(ROOT, 'b4_weights.pt'))
    SC.MEAN, SC.SV = 0.0050, 0.2089   # B4 training ground-truth statistics (see b4_full.log)
    Nx,Nt=128,1600; kl_basis(Nx,KMAX)
    slist=[0.40,0.50,0.60,0.70,0.80,0.90,1.00]
    AD=np.linspace(0.70,0.99,11); SD=np.linspace(0.30,1.10,33)
    PT=build_pt_eig(Nx,Nt,AD,SD)
    rng0=np.random.RandomState(777)                 # all s share the same batch of initial/coefficient fields; only s changes
    seeds0=rng0.randint(100000,size=NTE); xi0=rng0.randn(NTE,KMAX).astype(np.float32)
    ss=[]; eA=[]; eF=[]; ratio=[]
    for sv in slist:
        if round(sv,2) in done: continue
        seeds=seeds0
        a=np.full(NTE,0.85,np.float32); s=np.full(NTE,sv,np.float32); xi=xi0
        Ftr,U0,NU=true_field_batched(Nx,Nt,seeds,a,s,xi,chunk=NTE)
        with torch.no_grad():
            ea=err_of(fr,'A',Nx,U0,NU,a,s,Ftr,PT); ef=err_of(fno,'F',Nx,U0,NU,a,s,Ftr,PT)
        log('  s=%.2f  FrFNO=%6.3f%%  FNO=%6.3f%%  ratio R=%5.2f  K^{2s}=%5.2f'%(sv,ea,ef,ef/ea,100**sv))
        ss.append(sv); eA.append(ea); eF.append(ef); ratio.append(ef/ea)
        dump(exp2_s=np.array(ss),exp2_fr=np.array(eA),exp2_fno=np.array(eF),exp2_ratio=np.array(ratio))
    return np.array(ss),np.array(eA),np.array(eF),np.array(ratio)

def load_done():
    d1=set(); d2=set()
    if os.path.exists(OUT_NPZ):
        z=np.load(OUT_NPZ)
        if 'exp1_res' in z: d1=set(int(x-1) for x in z['exp1_res'])
        if 'exp2_s' in z: d2=set(round(float(x),2) for x in z['exp2_s'])
    return d1,d2

if __name__=='__main__':
    d1,d2=load_done()
    exp1(d1); exp2(d2)
    log('\nDone. Theoretical predictions: in Exp1 the error approaches a constant floor as N grows (neither diverges nor vanishes); in Exp2 logR is approximately linear in s with slope ~ln100=4.605.')
