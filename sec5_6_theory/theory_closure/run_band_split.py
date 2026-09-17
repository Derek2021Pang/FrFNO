# -*- coding: utf-8 -*-
"""Error decomposition on NONLINEAR Burgers:
- low-band (k<K): multiplier/approximation error
- high-band (k>=K): spectral truncation error
Compare FNO vs FrFNO on the nonlinear reference at 65^2 (N0=65, K=8)."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_SEC = os.path.dirname(_HERE); _ROOT = os.path.dirname(_SEC)
for p in (_HERE, _SEC, _ROOT):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, torch
from gpu_solver import dst2_torch
from frfno_legacy import DualNet, prop_at, ub_full_batch, DEV, DT
from theory_verify import build_pt_eig
import p0_platform as P
import frfno_legacy as LG
import surrogate_compare as SC

zr = dict(np.load(P.REF_NPZ, allow_pickle=True))
F513, seeds, a, s, xi = zr['F'], zr['seeds'], zr['a'], zr['s'], zr['xi']
Nx=64; K=8
Phi,e=P.get_phi(Nx)
U0=np.stack([LG.generate_multiscale_initial(Nx,8,int(sd)) for sd in seeds]).astype(np.float32)
NU=P.make_nu(Nx,xi,Phi,e)
Fds=P.downsample(F513,Nx).reshape(-1,Nx+1,Nx+1)
PT=build_pt_eig(Nx,3200,P.AD,P.SDG)
wpath=os.path.join(P.OUT,'kscan_N64_K8.pt')
ck=torch.load(wpath,map_location=DEV,weights_only=False)
MEAN=float(ck['MEAN']);SV=float(ck['SV']);pad=int(ck['pad']);SC.MEAN,SC.SV=MEAN,SV
net_a=DualNet(7,modes=K,pad=pad,field_dim=1,seed=1).to(DEV).float();net_a.load_state_dict(ck['FrFNO']);net_a.eval()
net_f=DualNet(2,modes=K,pad=pad,field_dim=0,seed=2).to(DEV).float();net_f.load_state_dict(ck['FNO']);net_f.eval()

ii,jj=np.meshgrid(np.arange(1,Nx+1),np.arange(1,Nx+1),indexing='ij')
kr=np.sqrt(ii**2+jj**2); low=kr<K; high=~low

def dst_energy(field):
    d=dst2_torch(torch.tensor(field[1:,1:],device=DEV,dtype=DT)[None])[0].cpu().numpy()
    return d
lo_f=[];hi_f=[];lo_a=[];hi_a=[];hi_u=[];hi_r=[]
for i in range(len(a)):
    aa,ss=float(a[i]),float(s[i])
    g=prop_at(aa,ss,*PT)
    with torch.no_grad():
        u0t=torch.tensor(U0[i:i+1],device=DEV,dtype=DT)
        nut=torch.tensor(NU[i:i+1],device=DEV,dtype=DT)
        xf=torch.stack([u0t,nut],-1); pf=MEAN+SV*net_f(xf,(aa,ss))
        gs=[prop_at(aa,ss,*PT)]+[prop_at(*z,*PT) for z in [(aa-.1,ss),(aa+.1,ss),(aa,ss-.08),(aa,ss+.08)]]
        ubs=[ub_full_batch(U0[i:i+1],gg,Nx) for gg in gs]
        xa=torch.stack([u0t,ubs[0],nut,ubs[1],ubs[2],ubs[3],ubs[4]],-1)
        pa=ubs[0]+SV*net_a(xa,(aa,ss))
    u_true=dst_energy(Fds[i]); f_pred=dst_energy(pf.cpu().numpy()[0]); a_pred=dst_energy(pa.cpu().numpy()[0])
    base=dst_energy(ubs[0].cpu().numpy()[0])
    # error per band
    lo_f.append(np.sqrt(np.sum(np.abs(f_pred[low]-u_true[low])**2)))
    hi_f.append(np.sqrt(np.sum(np.abs(f_pred[high]-u_true[high])**2)))
    lo_a.append(np.sqrt(np.sum(np.abs(a_pred[low]-u_true[low])**2)))
    hi_a.append(np.sqrt(np.sum(np.abs(a_pred[high]-u_true[high])**2)))
    hi_u.append(np.sqrt(np.sum(np.abs(u_true[high])**2)))  # truncation of u
    hi_r.append(np.sqrt(np.sum(np.abs((u_true-base)[high])**2)))  # truncation of r

print("=== Nonlinear Burgers, N0=65 K=8: error split by spectral band ===")
print(f"{'':12s} {'low k<K (multiplier)':>22s} {'high k>=K (truncation)':>24s}")
print(f"{'FNO':12s} {np.mean(lo_f):12.4f} {np.mean(hi_f):12.4f}")
print(f"{'FrFNO':12s} {np.mean(lo_a):12.4f} {np.mean(hi_a):12.4f}")
print(f"\nPure truncation (no network):")
print(f"  sigma_K(u)={np.mean(hi_u):.4f}   sigma_K(r)={np.mean(hi_r):.4f}")
print(f"\nLow-band improvement FNO->FrFNO: {np.mean(lo_f)/np.mean(lo_a):.2f}x")
print(f"High-band improvement FNO->FrFNO: {np.mean(hi_f)/np.mean(hi_a):.2f}x")
