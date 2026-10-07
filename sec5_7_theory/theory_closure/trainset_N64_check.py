# -*- coding: utf-8 -*-
"""Decisive check for trainset_N64.npz: recompute one (a,s) slice sample with the
CURRENT Gamma-fixed nl_solver and compare to the stored 10-05 training set.
Slice (i,j) = (4,4) (mid order pair), sample k = 0. Inputs U0[0], XI[4,4,0] are
rebuilt exactly as kscan_N64_train.build_trainset does."""
import os, sys, time
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sec5_7_theory", "theory_closure"))
from nl_solver import gpu_solve_nl_batch
from gpu_solver import spectral_eigvals_torch
from frfno_core import generate_multiscale_initial, A_GRID, S_GRID
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE

NX_TR, NL_NT, TN, BETA = 64, 400, 0.015, 3.0
z = np.load(os.path.join(ROOT, "theory_closure", "trainset_N64.npz"), allow_pickle=True)
U0_all, SOL_all, XI_all = z["U0"], z["SOL"], z["XI"]

i, j, k = 4, 4, 0
sol_stored = SOL_all[i, j, k]                       # (65,65)
xi = XI_all[i, j, k]                                # (KMAX,)
aa, ss = float(A_GRID[i]), float(S_GRID[j])
a_, s_ = np.array([aa], np.float32), np.array([ss], np.float32)

Phi, e = kl_basis(NX_TR, KMAX)
PHI_CACHE[(NX_TR, "Phi")], PHI_CACHE[(NX_TR, "e")] = Phi, e
U0 = np.stack([generate_multiscale_initial(NX_TR, 8, 1000 + k)]).astype(np.float32)
NU = make_nu(NX_TR, xi[None], Phi, e)
Q = spectral_eigvals_torch(NX_TR, 'cuda', torch.float32)
t0 = time.time()
fo = gpu_solve_nl_batch(torch.tensor(U0, device='cuda'), torch.tensor(NU, device='cuda'),
                        a_, s_, TN, NL_NT, NX_TR, beta=BETA, Q=Q, device='cuda',
                        dtype=torch.float32).cpu().numpy()[0]
el = time.time() - t0
d = np.abs(fo - sol_stored).max()
rel = np.linalg.norm(fo - sol_stored) / np.linalg.norm(sol_stored)
lines = [
    f"slice(i={i},j={j}) a={aa:.4f} s={ss:.4f} sample k={k} seed=1000+{k}",
    f"recompute {el:.1f}s; max|diff|={d:.6e}  relL2={rel:.6e}",
    f"U0 recon match: {np.abs(U0_all[k]-U0[0]).max():.2e}",
    f"max|stored|={np.abs(sol_stored).max():.4f}  mean|stored|={np.abs(sol_stored).mean():.4f}",
]
verdict = "一致（Γ 修复后生成）" if rel < 1e-3 else ("大偏差（疑似修复前旧训练集）" if rel > 1e-2 else "中等偏差，需进一步判断")
lines.append(f"verdict: {verdict}")
txt = "\n".join(lines)
print(txt)
with open(os.path.join(ROOT, "theory_closure", "trainset_N64_check_20261007.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
