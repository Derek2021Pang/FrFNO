# -*- coding: utf-8 -*-
"""Band-split error for B1 main weights (N0=17 K=10) at 17^2 and 129^2.
Low band k<K: multiplier/approximation; high band k>=K: truncation.
Uses frfno_core (field_dim=2) matching b1_weights.pt."""
import os, sys
ROOT = r'C:\Users\Dell\Desktop\FrFNO\FrFNO_repro20260916'
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,'sec5_2_B1_main'))
import numpy as np, torch
from frfno_core import DualNet, build_prop_table, prop_at, ub_full_batch, A_GRID, S_GRID, DEV, DT
from gpu_solver import dst2_torch
import frfno_core as FC
import b1_burgers as B1
from d_scan import kl_basis, KMAX, PHI_CACHE

MEAN, SV = 0.0051, 0.2266; DA, DSNB = 0.10, 0.08
W = torch.load(os.path.join(ROOT,'b1_weights.pt'), map_location=DEV, weights_only=False)
nets = {'FrFNO': DualNet(7, seed=1), 'FNO': DualNet(2, field_dim=2, seed=2)}
for k in nets: nets[k].load_state_dict(W[k]); nets[k].eval()

rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=B1.NTEST)
aa = rng.uniform(A_GRID.min(), A_GRID.max(), B1.NTEST).astype(np.float32)
ss = rng.uniform(S_GRID.min(), S_GRID.max(), B1.NTEST).astype(np.float32)
xi = rng.randn(B1.NTEST, KMAX).astype(np.float32)
K = 10

@torch.no_grad()
def pred(name, net, Nx, U0, NU, PT):
    out=[]
    for i in range(len(aa)):
        u0t=torch.tensor(U0[i:i+1],device=DEV,dtype=DT)
        nut=torch.tensor(NU[i:i+1],device=DEV,dtype=DT)
        a_,s_=float(aa[i]),float(ss[i])
        if name=='FrFNO':
            gs=[prop_at(a_,s_,*PT)]+[prop_at(*z,*PT) for z in [(a_-DA,s_),(a_+DA,s_),(a_,s_-DSNB),(a_,s_+DSNB)]]
            ubs=[ub_full_batch(U0[i:i+1],g,Nx) for g in gs]
            xb=torch.stack([u0t,ubs[0],nut,ubs[1],ubs[2],ubs[3],ubs[4]],-1)
            p=ubs[0].cpu().numpy()+SV*net(xb,(a_,s_)).cpu().numpy()
        else:
            xb=torch.stack([u0t,nut],-1)
            p=MEAN+SV*net(xb,(a_,s_)).cpu().numpy()
        out.append(p.reshape(Nx+1,Nx+1))
    return np.stack(out)

for Nx in (16, 128):
    Phi,e=kl_basis(Nx,KMAX); PHI_CACHE[(Nx,'Phi')],PHI_CACHE[(Nx,'e')]=Phi,e
    Nt = 400 if Nx==16 else 1600
    Ftr,U0,NU=B1.true_field_nl(Nx,Nt,seeds,aa,ss,xi,Phi,e)
    true=Ftr.reshape(-1,Nx+1,Nx+1)
    PT=build_prop_table(Nx,Nt=400)
    ii,jj=np.meshgrid(np.arange(1,Nx+1),np.arange(1,Nx+1),indexing='ij')
    kr=np.sqrt(ii**2+jj**2); low=kr<K; high=~low
    res={}
    for name in ('FrFNO','FNO'):
        P=pred(name,nets[name],Nx,U0,NU,PT)
        lo=[];hi=[]
        for i in range(len(aa)):
            td=dst2_torch(torch.tensor(true[i,1:,1:],device=DEV,dtype=DT)[None])[0].cpu().numpy()
            pd_=dst2_torch(torch.tensor(P[i,1:,1:],device=DEV,dtype=DT)[None])[0].cpu().numpy()
            lo.append(np.sqrt(np.sum(np.abs(pd_[low]-td[low])**2)))
            hi.append(np.sqrt(np.sum(np.abs(pd_[high]-td[high])**2)))
        res[name]=(np.mean(lo),np.mean(hi))
    print(f"\n=== Nx={Nx} ({(Nx+1)}^2), K={K} ===")
    print(f"{'':8s}{'low k<K':>12s}{'high k>=K':>12s}")
    print(f"{'FNO':8s}{res['FNO'][0]:12.4f}{res['FNO'][1]:12.4f}")
    print(f"{'FrFNO':8s}{res['FrFNO'][0]:12.4f}{res['FrFNO'][1]:12.4f}")
    print(f"low improvement: {res['FNO'][0]/res['FrFNO'][0]:.2f}x   high improvement: {res['FNO'][1]/res['FrFNO'][1]:.2f}x")
