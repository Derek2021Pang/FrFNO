# -*- coding: utf-8 -*-
"""Spectral truncation: sigma_K(u) and sigma_K(r) for dense K values (DST basis).

sigma_K(f)^2 = sum_{sqrt(i^2+j^2)>=K} |f_hat_ij|^2  (unnormalized DST tail energy)
- sigma_K(u): tail of true solution u  (what FNO must learn)
- sigma_K(r): tail of residual r=u-E0u0 (what FrFNO residual net must learn)
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_SEC = os.path.dirname(_HERE); _ROOT = os.path.dirname(_SEC)
for p in (_HERE, _SEC, _ROOT):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, torch
from gpu_solver import dst2_torch, spectral_eigvals_torch
from frfno_legacy import prop_at, ub_full_batch
from nl_solver import gpu_solve_nl_batch
from theory_verify import build_pt_eig
import p0_platform as P
import frfno_legacy as LG

zr = dict(np.load(P.REF_NPZ, allow_pickle=True))
F513, seeds, a, s, xi = zr['F'], zr['seeds'], zr['a'], zr['s'], zr['xi']

Nx = 64
Phi, e = P.get_phi(Nx)
U0 = np.stack([LG.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
NU = P.make_nu(Nx, xi, Phi, e)
Fds = P.downsample(F513, Nx).reshape(-1, Nx+1, Nx+1)
PT = build_pt_eig(Nx, 3200, P.AD, P.SDG)

# DST mode grid
ii, jj = np.meshgrid(np.arange(1, Nx+1), np.arange(1, Nx+1), indexing='ij')
kr = np.sqrt(ii**2 + jj**2)
Ks = list(range(2, 41, 2))  # dense K: 2,4,...,40

su_all = {K: [] for K in Ks}
sr_all = {K: [] for K in Ks}
for i in range(len(a)):
    aa, ss = float(a[i]), float(s[i])
    g = prop_at(aa, ss, *PT)
    base = ub_full_batch(U0[i:i+1], g, Nx).cpu().numpy()[0]
    u_ref = Fds[i]
    r = u_ref - base
    u_int = torch.tensor(u_ref[1:,1:], device=P.DEV, dtype=P.DT).unsqueeze(0)
    r_int = torch.tensor(r[1:,1:], device=P.DEV, dtype=P.DT).unsqueeze(0)
    u_dst = dst2_torch(u_int)[0].cpu().numpy()
    r_dst = dst2_torch(r_int)[0].cpu().numpy()
    pu = u_dst**2; pr = r_dst**2
    for K in Ks:
        mask = kr >= K
        su_all[K].append(np.sqrt(pu[mask].sum()))
        sr_all[K].append(np.sqrt(pr[mask].sum()))

su = np.array([np.mean(su_all[K]) for K in Ks])
sr = np.array([np.mean(sr_all[K]) for K in Ks])

print("K     sigma_K(u)    sigma_K(r)    ratio r/u")
for i, K in enumerate(Ks):
    print(f"{K:3d}  {su[i]:.6e}  {sr[i]:.6e}  {sr[i]/su[i]:.4f}")

# slopes over different ranges
for lo, hi in [(2,12), (4,16), (8,24), (12,40)]:
    idx = [i for i,K in enumerate(Ks) if lo<=K<=hi]
    if len(idx) >= 3:
        sl_u, _ = np.polyfit(np.log(np.array(Ks)[idx]), np.log(su[idx]), 1)
        sl_r, _ = np.polyfit(np.log(np.array(Ks)[idx]), np.log(sr[idx]), 1)
        sl_ratio, _ = np.polyfit(np.log(np.array(Ks)[idx]), np.log(sr[idx]/su[idx]), 1)
        print(f"K={lo}-{hi}: slope_u={sl_u:.3f}  slope_r={sl_r:.3f}  slope_ratio={sl_ratio:.3f}")

# Save
np.savez(os.path.join(P.OUT, 'spectral_truncation.npz'),
         Ks=np.array(Ks), sigma_u=su, sigma_r=sr,
         s_mean=float(np.mean(s)))
print(f"\nSaved. s_mean={np.mean(s):.3f}, theory ratio slope = -2s = {-2*np.mean(s):.3f}")
