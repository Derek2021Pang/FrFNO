# -*- coding: utf-8 -*-
"""Table-shift test: two big time steps, 0->0.015->0.03, one 400-step table.

The user's design: build ONE table covering t=0..0.03 with 400 sub-steps.
Big step 1 (0->0.015) queries the first 200 sub-steps (kernel G1).
Big step 2 (0.015->0.03) SHOULD query the second 200 sub-steps (kernel G2),
NOT replay the first-200 kernel on u(0.015).

For the linear modal problem u(t) = E_alpha((lambda t)^alpha) u0:
  G1(z) = E_alpha(z*0.015^alpha)   (kernel of the first 200 sub-steps)
  G2(z) = E_alpha(z*0.03^alpha) / E_alpha(z*0.015^alpha)
          (equivalent kernel of the second 200 sub-steps, memory folded in)
Ref u(0.03) = E_alpha(z*0.03^alpha) u0.
Plan A (correct, second-half kernel): uA = G2 * u(0.015), u(0.015)=G1*u0.
Plan B (what the current rollout code does, first-half kernel replayed):
        uB = G1 * u(0.015).
Expected: errA ~ 0 (machine precision); errB = semigroup violation
(16% at alpha=0.75, 39% at alpha=0.55, ~0 at alpha=1).
"""
import os, numpy as np, torch, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frfno_core import _l1_b, DEV, DT, ROOT

OUT = os.path.join(ROOT, 'table_shift_test_result.txt')
L = []
def log(s):
    print(s, flush=True); L.append(str(s))
    open(OUT, 'w', encoding='utf-8').write('\n'.join(L))

ZMIN, ZMAX, M = 1.0e-3, 5.0e5, 1600
T_BIG = 0.03      # two big steps of 0.015
Nt = 400          # 400 sub-steps: first 200 = 0->0.015, second 200 = 0.015->0.03
zg = np.logspace(np.log10(ZMIN), np.log10(ZMAX), M, dtype=np.float32)
zt = torch.tensor(zg, device=DEV, dtype=DT)

rng = np.random.RandomState(2024)
u0 = (rng.rand(M).astype(np.float64) + 0.01)   # arbitrary positive modal weights

log('# Table-shift test: two big steps 0->0.015->0.03 on a single 400-step table.')
log('# err = ||u_plan - u_ref|| / ||u_ref|| over the z-grid (z = lambda^s).')
log('# Plan A queries the SECOND-200 sub-step kernel G2 = E(0.03^a)/E(0.015^a).')
log('# Plan B replays the FIRST-200 sub-step kernel G1 = E(0.015^a) on u(0.015),')
log('#      which is exactly what the current rollout code does.')
log('%-8s %12s %12s' % ('alpha', 'PlanA(2nd200)', 'PlanB(1st200)'))

for al0 in (0.35, 0.55, 0.75, 0.95, 1.00):
    al = float(al0)
    b = torch.tensor(_l1_b(al, Nt), device=DEV, dtype=DT)
    dta = (T_BIG / Nt) ** al; b0 = b[0]
    hist = torch.empty((Nt, M), device=DEV, dtype=DT)
    hist[0] = torch.ones(M, device=DEV, dtype=DT); v = hist[0]
    for k in range(1, Nt + 1):
        rhs = b[k - 1] * hist[0]
        if k > 1:
            jj = torch.arange(1, k, device=DEV)
            rhs = rhs + torch.einsum('j,jm->m', b[k - jj - 1] - b[k - jj], hist[1:k])
        v = rhs / (b0 + dta * zt)
        if k < Nt: hist[k] = v
    h200 = hist[200].cpu().numpy().astype(np.float64)   # E(z*0.015^a): kernel G1
    h400 = v.cpu().numpy().astype(np.float64)           # E(z*0.03^a)
    G1 = h200
    G2 = h400 / np.maximum(h200, 1e-30)                 # second-200 equivalent kernel
    u015 = G1 * u0                                      # big step 1 (correct)
    uA = G2 * u015                                      # Plan A: second-half kernel
    uB = G1 * u015                                      # Plan B: first-half kernel replayed
    uRef = h400 * u0
    errA = np.linalg.norm(uA - uRef) / (np.linalg.norm(uRef) + 1e-12)
    errB = np.linalg.norm(uB - uRef) / (np.linalg.norm(uRef) + 1e-12)
    log('%-8s %11.4f%% %11.4f%%' % (('%.2f' % al), 100 * errA, 100 * errB))

log('DONE')
