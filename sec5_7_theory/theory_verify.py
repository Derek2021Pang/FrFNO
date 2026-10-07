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
from frfno_core import DualNet, DEV, DT, prop_at, build_prop_table
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
    """Propagator table with table points placed exactly at the grid's modal values
    lam=q^s (as in B4's build_pt_b4), so there is zero interpolation error in the
    modal direction at any s. The previous uniform-in-lam grid (M points over
    [0, qmax^s]) loses low-mode resolution catastrophically as s grows (the step
    grows like qmax^s/M), which produced the spurious FrFNO blow-up at s=0.9/1.0
    in Exp.2. Returns (AD,SD,PT[na,ns,n,n]) compatible with prop_at.

    Memory layout: the history is stored time-major (modal*ns rows x Nt columns)
    so the Volterra convolution einsum reduces along the contiguous time axis
    instead of across huge modal strides. This is the same recurrence, but runs
    ~5-10x faster at coarse grids (257^2 table: ~10 min instead of ~60 min)."""
    Q = spectral_eigvals(Nx); n = Nx-1
    AD = np.asarray(AD, np.float32); SD = np.asarray(SD, np.float32); ns = len(SD)
    qf = torch.tensor(Q.reshape(-1), device=DEV, dtype=DT)
    # lam over (ns, n*n), then flattened time-major together with the modal axis
    lam = (qf[None] ** torch.tensor(SD, device=DEV, dtype=DT)[:, None]).reshape(-1)  # (ns*n*n,)
    Mrow = lam.numel()
    PT = np.zeros((len(AD), ns, n, n), np.float32)
    for ia, al in enumerate(AD):
        b = torch.tensor(_l1_b(float(al), Nt), device=DEV, dtype=DT)
        dta = (TN/Nt)**float(al); b0 = b[0]
        den = b0 + dta*lam                                  # (Mrow,)
        H = torch.empty((Mrow, Nt), device=DEV, dtype=DT); H[:, 0] = 1.0
        v = H[:, 0].clone()
        for k in range(1, Nt+1):
            rhs = b[k-1]
            if k > 1:
                jj = torch.arange(1, k, device=DEV)
                rhs = rhs + torch.einsum('j,mj->m', b[k-jj-1]-b[k-jj], H[:, 1:k])
            v = rhs/den
            if k < Nt: H[:, k] = v
        PT[ia] = v.reshape(ns, n, n).cpu().numpy(); del H
    torch.cuda.empty_cache()
    return AD, SD, PT

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
    f=DualNet(2,field_dim=2,seed=tagF).to(DEV).float(); f.load_state_dict(W['FNO']); f.eval()
    return a,f

def err_of(net,kind,Nx,U0,NU,a,s,Ftr,PT):
    em,_,_=SC.eval_spectral(net,kind,Nx,U0,NU,a,s,Ftr,PT); return em

def exp1(done):
    log('\n########## Exp1 error floor (fixed K=10, B1 weights, matched dt refinement) ##########')
    fr,fno=load_pair(os.path.join(ROOT, 'b1_weights.pt'))
    SC.MEAN, SC.SV = 0.0052, 0.2338   # B1 training ground-truth statistics after Gamma(2-alpha) fix
    RES=[(16,400),(64,800),(128,1600),(256,3200)]
    rng=np.random.RandomState(2024); seeds=rng.randint(100000,size=NTE)
    a=rng.uniform(0.55,0.95,NTE).astype(np.float32); s=rng.uniform(0.35,0.78,NTE).astype(np.float32)
    xi=rng.randn(NTE,KMAX).astype(np.float32)
    AD=np.linspace(0.45,1.05,21); SD=np.linspace(0.25,0.90,21)
    res=[]; eA=[]; eF=[]
    for Nx,Nt in RES:
        if Nx in done: continue
        t0=time.time(); PT=build_prop_table(Nx, Nt=400); tpt=time.time()-t0   # Plan A: unified 1-D log table
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
    SC.MEAN, SC.SV = 0.0051, 0.2143   # B4 training ground-truth statistics after Gamma(2-alpha) fix (b4_rerun_20261005.log)
    Nx,Nt=128,1600; kl_basis(Nx,KMAX)
    slist=[0.40,0.50,0.60,0.70,0.80,0.90,1.00]
    AD=np.linspace(0.70,0.99,11); SD=np.linspace(0.30,1.10,33)
    PT=build_prop_table(Nx, Nt=400)   # Plan A: unified 1-D log table (same as training)
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
