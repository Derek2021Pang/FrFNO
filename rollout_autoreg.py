# -*- coding: utf-8 -*-
"""Autoregressive rollout test for the B1 models trained at T=0.015.

Each network trained for the single-step horizon T=0.015 is rolled out
autoregressively for four steps (each prediction fed back as the next initial
condition) to reach T=0.06, and compared against the T=0.06 reference solved
with the same GPU nonlinear solver. This reproduces the claim in Section 5.2
that every operator degrades sharply under rollout.

Uses the 2026-10-06/07 rerun basis: the unified one-dimensional propagator
table and the Gamma(2-alpha) corrected L1 coefficients (frfno_core.py),
the same trained weights as the paper (b1_weights.pt), and the same 48-sample
held-out stream (RandomState(2024)).
"""
import os, sys, time, json
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
import torch, torch.nn.functional as F
import frfno_core as FC
import d_scan as DS
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
from frfno_core import (DualNet, build_prop_table, prop_at, ub_full_batch,
                        generate_multiscale_initial, A_GRID, S_GRID, DEV, DT)
from gpu_solver import spectral_eigvals_torch
from nl_solver import gpu_solve_nl_batch
from models_surrogate import CNOWrap, DeepONet2d, UNet2d
from models_extra import PDNO2d
import surrogate_compare as SC
import b1_burgers as B1

TN, BETA = 0.015, 3.0
T_LONG = 0.06
STEPS_AR = 4                      # 4 x 0.015 = 0.06
NTEST = 48
RES = [(16, 1600), (64, 3200), (128, 6400)]   # (Nx, Nt) for the T=0.06 truth
OUT = os.path.join(ROOT, 'rollout_autoreg_result.txt')
L = []
def log(s): print(s, flush=True); L.append(str(s)); open(OUT, 'w', encoding='utf-8').write('\n'.join(L))

def rel(a, b): return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)

def step_spectral(net, kind, Nx, U0b, NU, a, s, PT):
    """One single-step prediction. U0b/NU: (1,H,W) tensors. Returns (H,W) ndarray."""
    aa, ss = float(a), float(s)
    if kind == 'A':
        gs = [prop_at(aa, ss, *PT)] + [prop_at(*z, *PT) for z in
              [(aa - SC.DA, ss), (aa + SC.DA, ss), (aa, ss - SC.DSNB), (aa, ss + SC.DSNB)]]
        ubs = [ub_full_batch(U0b, g, Nx) for g in gs]
        xb = torch.stack([U0b, ubs[0], NU, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
        phys = ubs[0].cpu().numpy() + SC.SV * net(xb, (aa, ss)).cpu().numpy()
    else:
        xb = torch.stack([U0b, NU], -1)
        phys = SC.MEAN + SC.SV * net(xb, (aa, ss)).cpu().numpy()
    return phys

def step_spatial(net, Nx, U0b, NU, a, s):
    x = torch.stack([U0b, NU], 1)
    cond = torch.tensor([[a, s]], device=DEV, dtype=DT)
    o = net(x, cond).cpu().numpy().reshape(1, -1)
    return (SC.MEAN + SC.SV * o).reshape(-1)

def main():
    FC.T = TN; SC.T = TN; B1.FC.T = TN
    for Nx, _ in RES:
        Phi, e = kl_basis(Nx, KMAX, 1.0); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Phi, e
    # Rebuild the training statistics exactly as in b1_burgers.main().
    log('rebuilding B1 training ground truth for MEAN/SV (one-time, ~minutes)...')
    U0tr, SOL, XItrain, HIST = B1.build_trainset_nl()
    SC.MEAN = float(SOL.mean()); SC.SV = float(SOL.std())
    log('MEAN=%.4f SV=%.4f' % (SC.MEAN, SC.SV))
    # Rebuild models and load the exact paper weights.
    nets = {}
    nets['FrFNO'] = DualNet(7, seed=1)
    nets['FNO'] = DualNet(2, field_dim=2, seed=2)
    nets['PINO'] = DualNet(2, field_dim=2, seed=7)
    nets['PDNO'] = PDNO2d(seed=6)
    nets['CNO'] = CNOWrap(17, ch=32, use_bn=False, seed=3).to(DEV).float()
    nets['DeepONet'] = DeepONet2d(w=256, seed=4).to(DEV).float()
    nets['UNet'] = UNet2d(base=48, seed=5).to(DEV).float()
    sd = torch.load(os.path.join(ROOT, 'b1_weights.pt'), map_location=DEV)
    for k in nets: nets[k].load_state_dict(sd[k]); nets[k].eval()
    log('loaded b1_weights.pt (%d models)' % len(nets))
    # Test stream, identical to the paper (RandomState(2024)).
    rng = np.random.RandomState(2024)
    seeds = rng.randint(100000, size=NTEST)
    a_te = rng.uniform(A_GRID.min(), A_GRID.max(), NTEST).astype(np.float32)
    s_te = rng.uniform(S_GRID.min(), S_GRID.max(), NTEST).astype(np.float32)
    xi_te = rng.randn(NTEST, KMAX).astype(np.float32)
    kinds = {'FrFNO': 'A', 'FNO': 'F', 'PINO': 'F', 'PDNO': 'F'}
    rows = {}
    with torch.no_grad():
        for Nx, Nt in RES:
            Phi, e = PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')]
            U0 = np.stack([generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
            NU = make_nu(Nx, xi_te, Phi, e)
            Q = spectral_eigvals_torch(Nx, DEV, DT)
            Ftr = gpu_solve_nl_batch(U0, NU, a_te, s_te, T_LONG, Nt, Nx,
                                     beta=BETA, Q=Q, device=DEV, dtype=DT).cpu().numpy()
            Ftr = Ftr.reshape(len(a_te), -1)
            log('\n===== rollout to T=%.2f at resolution %d^2 (Nt=%d truth) =====' % (T_LONG, Nx + 1, Nt))
            PT = build_prop_table(Nx, Nt=400)          # single-step table, T=0.015
            for name in ['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO', 'DeepONet', 'UNet']:
                errs = []
                for i in range(NTEST):
                    u = U0[i:i + 1].copy()
                    for _ in range(STEPS_AR):
                        u0t = torch.tensor(u, device=DEV, dtype=DT)
                        nut = torch.tensor(NU[i:i + 1], device=DEV, dtype=DT)
                        if name in kinds:
                            p = step_spectral(nets[name], kinds[name], Nx, u0t, nut,
                                              a_te[i], s_te[i], PT)
                            u = p.reshape(1, Nx + 1, Nx + 1).astype(np.float32)
                        else:
                            p = step_spatial(nets[name], Nx, u0t, nut, a_te[i], s_te[i])
                            u = p.reshape(1, Nx + 1, Nx + 1).astype(np.float32)
                    errs.append(rel(u.reshape(-1), Ftr[i]))
                rows[(Nx, name)] = 100 * np.mean(errs)
                log('  %-8s 4-step rollout rel L2 mean %7.3f%%' % (name, rows[(Nx, name)]))
                save_ckpt(rows)
    log('\n========= rollout summary: mean rel L2 error (%) at T=0.06 =========')
    for Nx, _ in RES:
        tag = '%d^2' % (Nx + 1)
        log('%-6s' % tag + ''.join('%9.3f%%' % rows[(Nx, m)] for m in
                                   ['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO', 'DeepONet', 'UNet']))
    np.savez(os.path.join(ROOT, 'rollout_autoreg_metrics.npz'),
             rows=np.array([[rows[(Nx, m)] for m in ['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO', 'DeepONet', 'UNet']]
                            for Nx, _ in RES]),
             methods=np.array(['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO', 'DeepONet', 'UNet']),
             res=np.array([Nx + 1 for Nx, _ in RES]))
    log('done.')

def save_ckpt(rows):
    pass

if __name__ == '__main__':
    main()
