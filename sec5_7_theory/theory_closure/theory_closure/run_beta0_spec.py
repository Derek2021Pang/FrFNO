# -*- coding: utf-8 -*-
"""Run beta=0 linear (variable-coefficient) reference solutions and measure DST tail."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_SEC = os.path.dirname(_HERE); _ROOT = os.path.dirname(_SEC)
for p in (_HERE, _SEC, _ROOT):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, torch
from gpu_solver import dst2_torch
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
PT = build_pt_eig(Nx, 3200, P.AD, P.SDG)

T = P.TN; Nt = 3200
print(f"Running beta=0 linear solve: Nx={Nx}, Nt={Nt}, n={len(seeds)}", flush=True)
from gpu_solver import dst2_torch, spectral_eigvals_torch
Q_gpu = spectral_eigvals_torch(Nx, P.DEV, P.DT)
u_lin_t = gpu_solve_nl_batch(
    torch.tensor(U0, device=P.DEV, dtype=P.DT),
    torch.tensor(NU, device=P.DEV, dtype=P.DT),
    a.astype(np.float64), s.astype(np.float64),
    T, Nt, Nx, beta=0.0, Q=Q_gpu, device=P.DEV, dtype=P.DT
)
u_lin = u_lin_t.cpu().numpy()
print(f"u_lin shape: {u_lin.shape}", flush=True)

ii, jj = np.meshgrid(np.arange(1, Nx+1), np.arange(1, Nx+1), indexing='ij')
kr = np.sqrt(ii**2 + jj**2)
Ks = [4, 6, 8, 12]

sr_all = {K: [] for K in Ks}
su_all = {K: [] for K in Ks}
for i in range(len(a)):
    aa, ss = float(a[i]), float(s[i])
    g = prop_at(aa, ss, *PT)
    base = ub_full_batch(U0[i:i+1], g, Nx).cpu().numpy()[0]
    u_ref = u_lin[i]
    r = u_ref - base
    r_int = torch.tensor(r[1:,1:], device=P.DEV, dtype=P.DT).unsqueeze(0)
    u_int = torch.tensor(u_ref[1:,1:], device=P.DEV, dtype=P.DT).unsqueeze(0)
    r_dst = dst2_torch(r_int)[0].cpu().numpy()
    u_dst = dst2_torch(u_int)[0].cpu().numpy()
    pr = r_dst**2; pu = u_dst**2
    for K in Ks:
        mask = kr >= K
        sr_all[K].append(np.sqrt(pr[mask].sum()))
        su_all[K].append(np.sqrt(pu[mask].sum()))

print("\n=== beta=0 (linear variable-coefficient) DST tail ===")
for K in Ks:
    sr = np.array(sr_all[K]); su = np.array(su_all[K])
    print(f"K={K:3d}: sigma_r={sr.mean():.4e}  sigma_u={su.mean():.4e}  ratio={(sr/su).mean():.4e}")

sr_arr = np.array([np.mean(sr_all[K]) for K in Ks])
su_arr = np.array([np.mean(su_all[K]) for K in Ks])
ratio_arr = sr_arr / su_arr
sr_slope, _ = np.polyfit(np.log(Ks), np.log(sr_arr), 1)
su_slope, _ = np.polyfit(np.log(Ks), np.log(su_arr), 1)
ratio_slope, _ = np.polyfit(np.log(Ks), np.log(ratio_arr), 1)
print(f"\nslope_r={sr_slope:.3f}  slope_u={su_slope:.3f}  slope_ratio={ratio_slope:.3f}")
print(f"theory: slope_ratio = -2s = {-2*0.582:.3f}")
