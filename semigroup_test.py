# -*- coding: utf-8 -*-
"""Direct semigroup test of the fractional propagator table (pure linear, no
network, no truth solves). For the integer order alpha=1 the Mittag-Leffler
semigroup is exact, E((t+tau)^alpha L) = E(t^alpha L) E(tau^alpha L), so
applying the single-step table twice must equal one application of the
two-step table. For 0<alpha<1 the Caputo memory breaks this identity.

Measure  || E(2tau) u0  -  E(tau) E(tau) u0 || / || E(2tau) u0 ||
over the paper's alpha/s range and at alpha=1. Seconds on GPU.
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import frfno_core as FC
from frfno_core import build_prop_table, prop_at, ub_full_batch, DEV, DT
import d_scan as DS
from d_scan import kl_basis, KMAX, PHI_CACHE

TN = 0.015
Nx = 16                                   # training resolution 17^2
Phi, e = kl_basis(Nx, KMAX, 1.0); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Phi, e
rng = np.random.RandomState(2024)
seeds = rng.randint(100000, size=12)
U0 = np.stack([DS.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)

def app(PT, alpha, s, U):
    g = prop_at(alpha, s, *PT)
    return ub_full_batch(torch.tensor(U, device=DEV, dtype=DT), g, Nx).cpu().numpy()

rows = []
rows.append('# Semigroup test of the 1-D propagator table at 17^2.')
rows.append('# err = ||E(2tau)u0 - E(tau)E(tau)u0|| / ||E(2tau)u0||  (mean over 12 fields, s=0.5).')
for alpha in [0.35, 0.55, 0.75, 0.95, 1.0]:
    s = 0.5
    FC.T = TN;       PT1 = build_prop_table(Nx, Nt=400)      # E(tau),   tau=0.015
    FC.T = 2 * TN;   PT2 = build_prop_table(Nx, Nt=800)      # E(2tau), tau=0.030
    FC.T = TN
    errs = []
    for i in range(U0.shape[0]):
        u = U0[i:i + 1]
        ua = app(PT2, alpha, s, u)                    # E(2tau)u0
        ub = app(PT1, alpha, s, app(PT1, alpha, s, u))  # E(tau)E(tau)u0
        errs.append(np.linalg.norm(ub - ua) / (np.linalg.norm(ua) + 1e-12))
    rows.append('  alpha=%.2f  semigroup violation  %6.3f%%' % (alpha, 100 * np.mean(errs)))
OUT = os.path.join(ROOT, 'semigroup_test_result.txt')
open(OUT, 'w', encoding='utf-8').write('\n'.join(rows))
print('\n'.join(rows))
print('saved', OUT)
