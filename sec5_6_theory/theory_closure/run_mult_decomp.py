# -*- coding: utf-8 -*-
"""Pure linear constant-coeff: FNO vs FrFNO error against exact E0*u0.
Isolates spectral-multiplier approximation error (N0=65, K=8)."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_SEC = os.path.dirname(_HERE); _ROOT = os.path.dirname(_SEC)
for p in (_HERE, _SEC, _ROOT):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, torch
from frfno_legacy import DualNet, prop_at, ub_full_batch, DEV, DT
from theory_verify import build_pt_eig
import p0_platform as P
import frfno_legacy as LG
import surrogate_compare as SC

zr = dict(np.load(P.REF_NPZ, allow_pickle=True))
seeds, a, s = zr['seeds'], zr['a'], zr['s']

Nx = 64; K = 8
U0 = np.stack([LG.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
PT = build_pt_eig(Nx, 3200, P.AD, P.SDG)

wpath = os.path.join(P.OUT, 'kscan_N64_K8.pt')
ck = torch.load(wpath, map_location=DEV, weights_only=False)
MEAN=float(ck['MEAN']); SV=float(ck['SV']); pad=int(ck['pad']); SC.MEAN,SC.SV=MEAN,SV
net_a = DualNet(7, modes=K, pad=pad, field_dim=1, seed=1).to(DEV).float()
net_a.load_state_dict(ck['FrFNO']); net_a.eval()
net_f = DualNet(2, modes=K, pad=pad, field_dim=0, seed=2).to(DEV).float()
net_f.load_state_dict(ck['FNO']); net_f.eval()

ef_list=[]; ea_list=[]
for i in range(len(a)):
    aa,ss=float(a[i]),float(s[i])
    g=prop_at(aa,ss,*PT)
    with torch.no_grad():
        base=ub_full_batch(U0[i:i+1],g,Nx)
        u0t=torch.tensor(U0[i:i+1],device=DEV,dtype=DT)
        nu1=torch.ones_like(u0t)
        xf=torch.stack([u0t,nu1],-1)
        pf=MEAN+SV*net_f(xf,(aa,ss))
        gs=[prop_at(aa,ss,*PT)]+[prop_at(*z,*PT) for z in [(aa-.1,ss),(aa+.1,ss),(aa,ss-.08),(aa,ss+.08)]]
        ubs=[ub_full_batch(U0[i:i+1],gg,Nx) for gg in gs]
        xa=torch.stack([u0t,ubs[0],nu1,ubs[1],ubs[2],ubs[3],ubs[4]],-1)
        pa=ubs[0]+SV*net_a(xa,(aa,ss))
        bn=base.cpu().numpy()[0,1:,1:]
        ef_list.append(np.linalg.norm(pf.cpu().numpy()[0,1:,1:]-bn)/np.linalg.norm(bn))
        ea_list.append(np.linalg.norm(pa.cpu().numpy()[0,1:,1:]-bn)/np.linalg.norm(bn))
ef=np.array(ef_list); ea=np.array(ea_list)
print("=== Pure linear (nu=1,beta=0), N0=65 K=8: rel L2 error vs exact E0*u0 ===")
print(f"FNO   : mean={ef.mean()*100:.2f}%  median={np.median(ef)*100:.2f}%")
print(f"FrFNO : mean={ea.mean()*100:.2f}%  median={np.median(ea)*100:.2f}%")
print(f"Ratio FNO/FrFNO = {ef.mean()/ea.mean():.1f}x")
