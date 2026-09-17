# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""C1-C4 rigor experiments (nonlinear fractional Burgers, T=.015, beta=3, train at 17^2, zero-shot super-resolution to 65/129).
C1 ablation: full / remove u_base (no_ub) / remove the dynamic fractional wavenumber (no_dyn) / remove the 4 neighbor orders (no_nb) / residual vs direct learning (direct)
C2 multiple seeds: FrFNO{1,11,21}, FNO{2,12,22}; report mean +/- std and worst case
C3 efficiency Pareto: training samples per grid point ntr in {16,32,64,128}; parameter count / wall time; super-resolution saves the high-resolution ground-truth cost (timed separately)
C4 sensitivity: order grids {3x3,5x5,9x9} / spectral modes in {6,10,14} (= theoretical K) / lr in {1e-3,1.5e-3,3e-3}
All use 3000 steps under one common protocol (ablations/sensitivities compare relative attributions, fairness first); the three test ground-truth tiers are built once and shared; incremental saving allows resume."""
import os, sys, time, json
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
import numpy as np, torch, torch.nn.functional as F
import frfno_core as FC
from frfno_core import (DualNet, build_prop_table, prop_at, ub_full_batch,
                           A_GRID, S_GRID, DEV, DT)
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
from gpu_solver import spectral_eigvals_torch
from models_surrogate import n_params
import surrogate_compare as SC
from b1_burgers import build_trainset_nl, true_field_nl, TN, BETA, NL_NT

FC.T=TN; SC.T=TN
FAST=os.environ.get('FAST')=='1'
STEPS=4 if FAST else 3000; NTE=2 if FAST else 24; RES=[(16,400),(64,800),(128,1600)]
DA,DSNB=0.10,0.08
OUT_TXT=os.path.join(ROOT, 'c_ablation_result')+('_fast' if FAST else '')+'.txt'
OUT_NPZ=os.path.join(ROOT, 'c_ablation')+('_fast' if FAST else '')+'.npz'
WPT=os.path.join(ROOT, 'c_weights')+('_fast' if FAST else '')+'.pt'
L=[]
def log(s): print(s,flush=True); L.append(str(s)); open(OUT_TXT,'w',encoding='utf-8').write('\n'.join(L))
def dump(d):
    z=dict(np.load(OUT_NPZ),allow_pickle=True) if os.path.exists(OUT_NPZ) else {}
    z.update(d); np.savez(OUT_NPZ,**z)
def done_keys():
    if not os.path.exists(OUT_NPZ): return set()
    z=np.load(OUT_NPZ,allow_pickle=True)
    return set(json.loads(z['done'].item())) if 'done' in z.files else set()
def save_done(k):
    z=dict(np.load(OUT_NPZ,allow_pickle=True)) if os.path.exists(OUT_NPZ) else {}
    s=set(json.loads(z['done'].item()) if 'done' in z else []); s.add(k); z['done']=json.dumps(sorted(s)); np.savez(OUT_NPZ,**z)

# ---------- Variant inputs/targets/reconstruction (training and evaluation use exactly the same pipeline) ----------
def ubs5(a,s,U0,Nx,PT):
    gs=[prop_at(a,s,*PT)]+[prop_at(*z,*PT) for z in[(a-DA,s),(a+DA,s),(a,s-DSNB),(a,s+DSNB)]]
    return [ub_full_batch(U0,g,Nx) for g in gs]

def build_io(variant,a,s,U0b,NUb,Nx,PT):
    """Returns (inputxb, central propagator ubs0 [full-solution baseline, used in the residual case]). U0b/NUb: (B,H,W) tensors."""
    if variant in ('full','no_dyn','direct'):
        ubs=ubs5(a,s,U0b,Nx,PT); u0=ubs[0]
        xb=torch.stack([U0b,ubs[0],NUb,ubs[1],ubs[2],ubs[3],ubs[4]],-1); ind=7
    elif variant=='no_nb':
        ubs=ubs5(a,s,U0b,Nx,PT); u0=ubs[0]
        xb=torch.stack([U0b,ubs[0],NUb],-1); ind=3
    elif variant=='no_ub':
        u0=None
        xb=torch.stack([U0b,NUb],-1); ind=2
    return xb,u0,ind

RESID={'full':1,'no_dyn':1,'no_nb':1,'no_ub':0,'direct':0}   # 1=learn the residual (add ubs0 back), 0=learn the full solution (MEAN baseline)
FD={'full':2,'no_dyn':0,'no_nb':2,'no_ub':2,'direct':2}

def train_var(cfg,SOL,U0tr,XI,PT16,MEAN,SV):
    var=cfg['v']; net=DualNet(cfg['ind'],modes=cfg['modes'],field_dim=cfg['fd'],seed=cfg['seed']).to(DEV).float()
    op=torch.optim.Adam(net.parameters(),cfg['lr'],weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(op,STEPS,5e-5); net.train()
    rng=np.random.RandomState(123); Phi,e=PHI_CACHE[(16,'Phi')],PHI_CACHE[(16,'e')]
    gi=cfg['grid']; ntr=min(cfg['ntr'],SOL.shape[2]); resid=RESID[var]; t0=time.time()
    for st in range(STEPS+1):
        ia=gi[rng.randint(len(gi))]; js=gi[rng.randint(len(gi))]
        itr=torch.randperm(ntr,device=DEV)[:64].cpu().numpy(); a,s=A_GRID[ia],S_GRID[js]
        sol=torch.tensor(SOL[ia,js,itr],device=DEV,dtype=DT)
        u0b=torch.tensor(U0tr[itr],device=DEV,dtype=DT)
        nub=torch.tensor(make_nu(16,XI[ia,js,itr],Phi,e),device=DEV,dtype=DT)
        xb,u0,_=build_io(var,a,s,u0b,nub,16,PT16)
        tg=(sol-u0)/SV if resid==1 else (sol-MEAN)/SV
        loss=F.mse_loss(net(xb,(a,s)),tg); op.zero_grad();loss.backward();op.step();sch.step()
    net.eval(); return net,time.time()-t0

@torch.no_grad()
def eval_var(net,cfg,Nx,U0,NU,a,s,Ftrue,PT,MEAN,SV):
    var=cfg['v']; resid=RESID[var]; errs=[]
    for i in range(len(a)):
        u0t=torch.tensor(U0[i:i+1],device=DEV,dtype=DT); nut=torch.tensor(NU[i:i+1],device=DEV,dtype=DT)
        aa,ss=float(a[i]),float(s[i]); xb,u0,_=build_io(var,aa,ss,u0t,nut,Nx,PT)
        out=SV*net(xb,(aa,ss)).cpu().numpy()
        phys=(u0.cpu().numpy()+out) if resid==1 else (MEAN+out)
        errs.append(np.linalg.norm(phys.reshape(-1)-Ftrue[i])/(np.linalg.norm(Ftrue[i])+1e-12))
    return 100*np.mean(errs),100*np.median(errs)

def cfg(key,v,seed=1,modes=10,lr=1.5e-3,ntr=128,grid=None,ind=None,fd=None):
    return dict(key=key,v=v,seed=seed,modes=modes,lr=lr,ntr=ntr,
                grid=np.arange(9) if grid is None else np.array(grid),
                ind=(7 if v in('full','no_dyn','direct') else 3 if v=='no_nb' else 2) if ind is None else ind,
                fd=FD[v] if fd is None else fd)

def main():
    for Nx,_ in RES:
        P,e=kl_basis(Nx,KMAX); PHI_CACHE[(Nx,'Phi')],PHI_CACHE[(Nx,'e')]=P,e
    log('building training ground truth...'); U0tr,SOL,XI,_=build_trainset_nl(); MEAN=float(SOL.mean()); SV=float(SOL.std())
    log('MEAN=%.4f SV=%.4f, building test ground truth (shared across three tiers, %d samples)...'%(MEAN,SV,NTE))
    PT={Nx:build_prop_table(Nx,Nt=Nt) for Nx,Nt in RES}
    rng=np.random.RandomState(2024); seeds=rng.randint(100000,size=NTE)
    TE={}
    for Nx,Nt in RES:
        ra=np.random.RandomState(2024); _=ra.randint(100000,size=NTE)
        at=rng.uniform(.55,.95,NTE).astype(np.float32); st_=rng.uniform(.35,.78,NTE).astype(np.float32)
        xi=rng.randn(NTE,KMAX).astype(np.float32)
        F,U,NU=true_field_nl(Nx,Nt,seeds,at,st_,xi,PHI_CACHE[(Nx,'Phi')],PHI_CACHE[(Nx,'e')])
        TE[Nx]=(U,NU,at,st_,F); log('  test ground truth N=%d^2 done'%(Nx+1))
    # Time the high-resolution ground-truth cost (C3): single-sample solve wall time at 17^2 vs 129^2
    t=time.time(); true_field_nl(16,400,seeds[:1],at[:1],st_[:1],np.random.randn(1,KMAX).astype(np.float32),*[PHI_CACHE[(16,k)] for k in('Phi','e')]); c17=time.time()-t
    t=time.time(); true_field_nl(128,1600,seeds[:1],at[:1],st_[:1],np.random.randn(1,KMAX).astype(np.float32),*[PHI_CACHE[(128,k)] for k in('Phi','e')]); c129=time.time()-t
    log('single-sample ground-truth wall time: 17^2=%.3fs 129^2=%.3fs ratio %.1fx'%(c17,c129,c129/c17)); dump(dict(cost17=c17,cost129=c129))

    configs=[]
    # C1 ablation (full FrFNO backbone, single varying factor)
    configs+= [cfg('C1_full','full'),cfg('C1_no_ub','no_ub'),cfg('C1_no_dyn','no_dyn'),
               cfg('C1_no_nb','no_nb'),cfg('C1_direct','direct')]
    # C2 multiple seeds
    configs+= [cfg('C2_Fr_s11','full',11),cfg('C2_Fr_s21','full',21),
               cfg('C2_FNO_s2','no_ub',2,ind=2,fd=0),cfg('C2_FNO_s12','no_ub',12,ind=2,fd=0),cfg('C2_FNO_s22','no_ub',22,ind=2,fd=0)]
    # C3 number of training samples (per grid point)
    configs+= [cfg('C3_n16','full',ntr=16),cfg('C3_n32','full',ntr=32),cfg('C3_n64','full',ntr=64)]
    # C4 sensitivity
    configs+= [cfg('C4_grid3','full',grid=[0,4,8]),cfg('C4_grid5','full',grid=[0,2,4,6,8]),
               cfg('C4_m6','full',modes=6),cfg('C4_m8','full',modes=8),
               cfg('C4_lr1e3','full',lr=1e-3),cfg('C4_lr3e3','full',lr=3e-3)]
    if FAST: configs=[c for c in configs if c['key'].startswith('C1')]
    done=done_keys(); W=torch.load(WPT,map_location=DEV) if os.path.exists(WPT) else {}
    log('%d configurations total, %d already done'%(len(configs),len(done)))
    for c in configs:
        if c['key'] in done: log('  skip (already done) '+c['key']); continue
        net,tr=train_var(c,SOL,U0tr,XI,PT[16],MEAN,SV); W[c['key']]=net.state_dict(); torch.save(W,WPT)
        row=dict(key=c['key'],v=c['v'],seed=c['seed'],modes=c['modes'],lr=c['lr'],ntr=c['ntr'],
                 grid=len(c['grid']),nparamK=round(n_params(net)/1e3,1),train_s=round(tr,1),e={})
        for Nx,_ in RES:
            U,NU,at,st_,F=TE[Nx]; em,emd=eval_var(net,c,Nx,U,NU,at,st_,F,PT[Nx],MEAN,SV); row['e'][Nx+1]=round(em,3)
        dump({c['key']:json.dumps(row)}); save_done(c['key'])
        log('  %-12s params %.1fK train %.0fs  17/65/129 = %.3f / %.3f / %.3f'
            %(c['key'],row['nparamK'],tr,row['e'][17],row['e'][65],row['e'][129]))
        del net; torch.cuda.empty_cache()
    log('\nall configurations done -> c_ablation.npz / c_weights.pt')

if __name__=='__main__': main()
