# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""B1 full-regression: FrFNO only, default config (NTR=128, STEPS=8000, NTEST=48).
Uses the new 1-D-table-backed build_prop_table. Compare against old headline numbers:
  17^2: 1.808%, 65^2: 7.493%, 129^2: 9.013% (relative L2).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
ROOT = _PKG_ROOT; sys.path.insert(0, ROOT)
import frfno_core as FC
FC.T = 0.015
from frfno_core import (DualNet, build_prop_table, prop_at, ub_full_batch,
                         A_GRID, S_GRID, DA, DS, NX_TR, DEV, DT, generate_multiscale_initial)
from fractional_pde_study.solver_spacetime_fractional import spectral_eigvals
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
from nl_solver import gpu_solve_nl_batch
from gpu_solver import spectral_eigvals_torch

NTR = 128; STEPS = 8000; NTEST = 48
TN, BETA, NL_NT = 0.015, 3.0, 400
RES = [(16, 400), (64, 800), (128, 1600)]
OUT = os.path.join(ROOT, 'b1_frfno_regression_result.txt')
L = []
def log(s):
    print(s, flush=True); L.append(str(s))
    try: open(OUT, 'w', encoding='utf-8').write('\n'.join(L))
    except Exception: pass
def sync():
    torch.cuda.synchronize() if DEV.type == 'cuda' else None

OLD = {16: 1.808, 64: 7.493, 128: 9.013}  # old headline relative L2 in %

# ---- training data ----
log('=== B1 FrFNO full regression (NTR=%d, STEPS=%d, NTEST=%d) ===' % (NTR, STEPS, NTEST))
Phi16, e16 = kl_basis(16, KMAX)
PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')] = Phi16, e16
rng = np.random.RandomState(0)
U0tr = np.stack([generate_multiscale_initial(16, 8, 1000 + i) for i in range(NTR)]).astype(np.float32)
XI = rng.randn(81, NTR, KMAX).astype(np.float32).reshape(9, 9, NTR, KMAX)
AA = np.repeat(A_GRID, 9 * NTR); SS = np.tile(np.repeat(S_GRID, NTR), 9)
NU_all = make_nu(16, XI.reshape(81 * NTR, KMAX), Phi16, e16)
NUtr = NU_all.reshape(9, 9, NTR, 17, 17)
UB = np.concatenate([U0tr] * 81, 0)
Qtr = spectral_eigvals_torch(16, DEV, DT)
sols = []; t0 = time.time()
for i0 in range(0, len(AA), 1024):
    fo = gpu_solve_nl_batch(UB[i0:i0 + 1024], NU_all[i0:i0 + 1024], AA[i0:i0 + 1024],
                             SS[i0:i0 + 1024], TN, NL_NT, 16, beta=BETA, Q=Qtr, device=DEV, dtype=DT)
    sols.append(fo.cpu().numpy())
SOL = np.concatenate(sols, 0).reshape(9, 9, NTR, 17, 17)
MEAN = float(SOL.mean()); SV = float(SOL.std())
log('training data: %d samples, MEAN=%.4f SV=%.4f, %.1fs' % (9 * 9 * NTR, MEAN, SV, time.time() - t0))

# ---- test data ----
rng2 = np.random.RandomState(2024)
seeds = rng2.randint(100000, size=NTEST)
a_te = rng2.uniform(A_GRID.min(), A_GRID.max(), NTEST).astype(np.float32)
s_te = rng2.uniform(S_GRID.min(), S_GRID.max(), NTEST).astype(np.float32)
xi_te = rng2.randn(NTEST, KMAX).astype(np.float32)
TEST = {}
for Nx, Nt in RES:
    if (Nx, 'Phi') not in PHI_CACHE:
        Ph, eh = kl_basis(Nx, KMAX); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Ph, eh
    Ph, eh = PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')]
    U0te = np.stack([generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
    NUte = make_nu(Nx, xi_te, Ph, eh)
    Q = spectral_eigvals_torch(Nx, DEV, DT)
    Ft = gpu_solve_nl_batch(U0te, NUte, a_te, s_te, TN, Nt, Nx, beta=BETA, Q=Q, device=DEV, dtype=DT).cpu().numpy()
    TEST[Nx] = (U0te, NUte, Ft)
    log('test data @%d^2 ready' % (Nx + 1))

# ---- build table (new 1-D cached + render) ----
t0 = time.time(); PT = {}
for Nx, _ in RES:
    PT[Nx] = build_prop_table(Nx, Nt=NL_NT)
sync(); t_build = time.time() - t0
log('build_prop_table (3 resolutions): %.2fs (1-D table cached, render per res)' % t_build)

# ---- train FrFNO ----
torch.manual_seed(1); np.random.seed(1)
net = DualNet(7, seed=1).to(DEV)
opt = torch.optim.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, 5e-5)
ADtr, SDtr, PTtr = PT[16]
net.train(); t0 = time.time()
for st in range(STEPS):
    itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
    ia = np.random.randint(9); js = np.random.randint(9)
    a, s = float(A_GRID[ia]), float(S_GRID[js])
    sol = torch.tensor(SOL[ia, js, itr], device=DEV, dtype=DT)
    u0b = torch.tensor(U0tr[itr], device=DEV, dtype=DT)
    nub = torch.tensor(NUtr[ia, js, itr], device=DEV, dtype=DT)
    gs = [prop_at(a, s, ADtr, SDtr, PTtr)] + [
        prop_at(*z, ADtr, SDtr, PTtr) for z in [(a - DA, s), (a + DA, s), (a, s - DS), (a, s + DS)]]
    ubs = [ub_full_batch(U0tr[itr], g, NX_TR) for g in gs]
    xb = torch.stack([u0b, ubs[0], nub, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
    tg = (sol - ubs[0]) / SV
    loss = F.mse_loss(net(xb, (a, s)), tg)
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if (st + 1) % 2000 == 0:
        log('  step %d/%d loss=%.5f' % (st + 1, STEPS, float(loss)))
net.eval(); sync(); t_train = time.time() - t0
log('training: %.1fs' % t_train)

# ---- eval ----
log('-' * 70)
log('%-12s %12s %12s %12s' % ('resolution', 'new (1-D)', 'old (2-D)', 'diff'))
for Nx, _ in RES:
    U0te, NUte, Ftrue = TEST[Nx]
    ADx, SDx, PTx = PT[Nx]
    t0 = time.time(); preds = []
    for m in range(NTEST):
        a, s = float(a_te[m]), float(s_te[m])
        gs = [prop_at(a, s, ADx, SDx, PTx)] + [
            prop_at(*z, ADx, SDx, PTx) for z in [(a - DA, s), (a + DA, s), (a, s - DS), (a, s + DS)]]
        ubs = [ub_full_batch(U0te[m:m + 1], g, Nx) for g in gs]
        u0t = torch.tensor(U0te[m:m + 1], device=DEV, dtype=DT)
        nut = torch.tensor(NUte[m:m + 1], device=DEV, dtype=DT)
        xb = torch.stack([u0t, ubs[0], nut, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
        with torch.no_grad(): r = net(xb, (a, s)).cpu().numpy()
        preds.append((ubs[0].cpu().numpy() + SV * r)[0])
    sync(); t_eval = time.time() - t0
    P = np.stack(preds)
    err = float(np.linalg.norm(P - Ftrue) / (np.linalg.norm(Ftrue) + 1e-12)) * 100
    old = OLD[Nx]
    log('%-12s %11.3f%% %11.3f%% %11.2e' % ('%d^2' % (Nx + 1), err, old, abs(err - old)))
log('-' * 70)
log('build=%.2fs train=%.1fs' % (t_build, t_train))
log('Regression check done. Compare "diff" column above against old headline numbers (1.808/7.493/9.013 %).')
log('saved -> %s' % OUT)
