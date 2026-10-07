# -*- coding: utf-8 -*-
"""Decisive check: recompute sample 0 of p0_ref513.npz with the CURRENT (Gamma-fixed)
nl_solver and compare to the stored 10-05 ground truth. If the stored field is
Gamma-fixed, max|diff| is at float32 rounding level; if it predates the fix, the
diff is ~10% (Gamma(2-alpha) ~ 0.89-0.97 for alpha in [0.55,0.95])."""
import os, sys, time
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from nl_solver import gpu_solve_nl_batch
from gpu_solver import spectral_eigvals_torch
import frfno_core as LG

z = np.load(os.path.join(ROOT, "theory_closure", "p0_ref513.npz"), allow_pickle=True)
F, U0, NU = z["F"], z["U0"], z["NU"]
a0, s0 = float(z["a"][0]), float(z["s"][0])
TN, NT_REF, NX_REF, BETA = 0.015, 6400, 512, 3.0

DEV = 'cuda'
DT = torch.float32
Q = spectral_eigvals_torch(NX_REF, DEV, DT)
u0b = torch.tensor(U0[0:1], device=DEV, dtype=DT)
nub = torch.tensor(NU[0:1], device=DEV, dtype=DT)
t0 = time.time()
f_rec = gpu_solve_nl_batch(u0b, nub, np.array([a0]), np.array([s0]), TN, NT_REF, NX_REF,
                           beta=BETA, Q=Q, device=DEV, dtype=DT).cpu().numpy()[0]
el = time.time() - t0
f_old = F[0]
d = np.abs(f_rec - f_old).max()
rel = np.linalg.norm(f_rec - f_old) / np.linalg.norm(f_old)
lines = [
    f"sample0: a={a0:.4f} s={s0:.4f} seed={z['seeds'][0]}",
    f"recompute {el:.1f}s; max|diff|={d:.6e}  relL2={rel:.6e}",
    f"max|stored|={np.abs(f_old).max():.4f}  mean|stored|={np.abs(f_old).mean():.4f}",
]
verdict = "一致（Γ 修复后生成）" if rel < 1e-3 else ("大偏差（疑似修复前旧真值）" if rel > 1e-2 else "中等偏差，需进一步判断")
lines.append(f"verdict: {verdict}")
txt = "\n".join(lines)
print(txt)
out = os.path.join(ROOT, "theory_closure", "p0_ref513_check_20261007.txt")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
