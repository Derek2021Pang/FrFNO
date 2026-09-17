# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SEC = _os.path.dirname(_HERE); _PKG_ROOT = _os.path.dirname(_SEC)
for _p in (_HERE, _SEC, _PKG_ROOT):
    if _p not in _sys.path: _sys.path.insert(0, _p)
"""Probe P1-4: the first-order Born explained fraction (vs eta), mode-by-mode overlap of the residual spectrum with the first-order response, and high-mode dominance."""
import sys, numpy as np, torch
import os
ROOT = _PKG_ROOT
sys.path.insert(0, os.path.join(ROOT, 'theory_closure'))
import theory_p15_p14 as M
from gpu_solver import spectral_eigvals_torch, dst2_torch

Nx, Nt, n, bsx = 192, 1600, 8, 8
a = 0.85
Q = spectral_eigvals_torch(Nx, M.DEV, M.DT)
n_in = Nx-1; R, radii = M.radial_bins(n_in); Ks = np.arange(2, n_in//2, 1)
w = M.fixed_w(Nx, seed=777, kmax=max(16, Nx//3), beta=1.5)

@torch.no_grad()
def run(p, T, s, eta):
    U0 = np.stack([M.powerlaw_u0(Nx, p=p, seed=500+i) for i in range(n)]).astype(np.float32)
    ub, u1 = M.solve_born_batch(U0, w, a, s, T, Nt, Nx, Q=Q, bs=bsx)
    ut = M.solve_c1_batch(U0, w, eta, a, s, T, Nt, Nx, Q=Q, bs=bsx)
    rf, r1 = ut-ub, eta*u1
    Pr=np.zeros(len(radii)); P1=np.zeros(len(radii)); Pb=np.zeros(len(radii))
    Tr=np.zeros(len(Ks)); Tb=np.zeros(len(Ks)); ex=[]
    for i in range(n):
        sp=lambda f: dst2_torch(torch.tensor(f[1:Nx,1:Nx][None],device=M.DEV,dtype=M.DT))[0].cpu().numpy()
        sr,s1,sb=sp(rf[i]),sp(r1[i]),sp(ub[i])
        Pr+=M.radial_power(sr,R,radii)/n; P1+=M.radial_power(s1,R,radii)/n; Pb+=M.radial_power(sb,R,radii)/n
        Tr+=M.cum_tail(sr,Ks)/n; Tb+=M.cum_tail(sb,Ks)/n; ex.append(M.rel(rf[i].ravel(),r1[i].ravel()))
    # mode-by-mode power overlap (effective dynamic range)
    m=(Pr>Pr.max()*1e-8)&(P1>0); spec_fit=np.median(Pr[m]/P1[m]); spec_logstd=np.std(np.log10(Pr[m]/P1[m]))
    iTr=np.interp(radii,Ks,Tr); iTb=np.interp(radii,Ks,Tb)
    v=(iTr>iTr.max()*1e-10)&(iTb>iTb.max()*1e-10)&(iTb>0); dom=np.sqrt(iTr[v]/iTb[v])
    return np.mean(ex), spec_fit, spec_logstd, float(dom[-1])

for p in [1.3, 2.0]:
    for s in [0.4,0.6,0.8]:
        row=[]
        for eta in [0.05,0.1,0.2,0.4]:
            ex,fit,ls,dom=run(p,0.005,s,eta); row.append(f"e{eta}={ex:.3f}")
        ex,fit,ls,dom=run(p,0.005,s,0.2)
        print(f"p={p} s={s}: explain[eta] {' '.join(row)} | Pr/Pr1 median={fit:.2f} logstd={ls:.2f} dom_end={dom:.2f}x",flush=True)
print('PROBE DONE')
