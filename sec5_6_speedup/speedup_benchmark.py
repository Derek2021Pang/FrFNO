# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
Surrogate speedup / break-even benchmark for FrFNO (B1: 2D space-time fractional Burgers).

Quantifies where the surrogate wins when the spectral fractional-Laplacian direct
solver is already fast:
  (A) direct solver wall-clock (linear / nonlinear, single query / batched amortized)
  (B) FrFNO online inference wall-clock (analytic base + conditional residual net)
  (C) one-time offline cost and break-even number of queries
  (D) total wall-clock vs number of queries and the resulting speedup

Hardware: single RTX 3090, float32. Timings are medians over repeated runs with
warm-up; CUDA is synchronized before/after every measurement.
"""
import os, sys, time, statistics
import numpy as np
import torch

sys.path.insert(0, _PKG_ROOT)
from frfno_core import (DualNet, build_prop_table, prop_at, ub_full_batch,
                        DEV, DT)
from gpu_solver import gpu_solve_batch, spectral_eigvals_torch
from nl_solver import gpu_solve_nl_batch

# ---------------- problem config (B1) ----------------
T = 0.015
A0, S0 = 0.75, 0.55
DA, DSNB = 0.10, 0.08
SV = 0.2266                       # training-set std from b1_burgers result
ROOT = _PKG_ROOT
CACHE = os.path.join(ROOT, 'cache')
OUT = os.path.join(ROOT, 'speedup_benchmark_result.txt')
L = []
def log(s):
    print(s, flush=True); L.append(str(s))
    open(OUT, 'w', encoding='utf-8').write('\n'.join(L))

def sync():
    if DEV.type == 'cuda': torch.cuda.synchronize()

def med_ms(fn, warm=2, reps=5):
    for _ in range(warm): fn()
    sync(); ts = []
    for _ in range(reps):
        sync(); t0 = time.time(); fn(); sync()
        ts.append((time.time() - t0) * 1000)
    return statistics.median(ts)

def rand_inputs(B, Nx, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    u0 = 0.3 * torch.randn(B, Nx + 1, Nx + 1, generator=g)
    nu = 1.0 + 0.2 * torch.rand(B, Nx + 1, Nx + 1, generator=g)
    return u0, nu

# ---------------- load trained FrFNO ----------------
net = DualNet(7, seed=1).eval()
w = torch.load(os.path.join(ROOT, 'b1_weights.pt'), map_location=DEV, weights_only=False)
net.load_state_dict(w['FrFNO'])
npar = sum(p.numel() for p in net.parameters())
log('Device: %s %s | FrFNO params %.1fK | trained b1 weights loaded'
    % (DEV, torch.cuda.get_device_name(0), npar / 1e3))

def get_PT(Nx, Nt):
    """Build propagator table with disk cache (offline one-time cost)."""
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, 'prop_Nx%d_Nt%d_na41_ns41.npz' % (Nx, Nt))
    if os.path.exists(fn):
        d = np.load(fn); return d['AD'], d['SD'], d['PT'], 0.0
    t0 = time.time()
    AD, SD, PT = build_prop_table(Nx, Nt=Nt)
    dt = time.time() - t0
    np.savez(fn, AD=AD, SD=SD, PT=PT)
    return AD, SD, PT, dt

@torch.no_grad()
def frfno_online(B, Nx, PTtuple, a=A0, s=S0):
    """B online queries: analytic base fields + conditional residual network."""
    AD, SD, PT = PTtuple
    u0, nu = rand_inputs(B, Nx)
    u0 = u0.to(DEV, DT); nu = nu.to(DEV, DT)
    gs = ([prop_at(a, s, AD, SD, PT)] +
          [prop_at(*z, AD, SD, PT) for z in
           [(a - DA, s), (a + DA, s), (a, s - DSNB), (a, s + DSNB)]])
    ub0_l, feats = [], []
    for i in range(B):
        ubs = [ub_full_batch(u0[i:i+1], g, Nx) for g in gs]
        ub0_l.append(ubs[0])
        feats.append(torch.stack([u0[i:i+1], ubs[0], nu[i:i+1], *ubs[1:]], -1))
    xb = torch.cat(feats, 0)
    out = []
    for i in range(B):
        out.append(ub0_l[i] + SV * net(xb[i:i+1], (a, s)))
    return torch.cat(out, 0)

# ================= (A) direct solver =================
log('\n================ (A) Direct solver wall-clock (GPU, ms) ================')
log('Resolution  Nt    linear B=1  nonlinear B=1  linear B=64/field  nonlinear B=64/field')
solver_t = {}
RES = [(16, 400), (32, 800), (64, 1600), (128, 3200)]
for Nx, Nt in RES:
    Q = spectral_eigvals_torch(Nx, DEV, DT)
    u0_1, nu_1 = rand_inputs(1, Nx)
    rep1 = {16: 8, 32: 5, 64: 3, 128: 2}[Nx]
    lin1 = med_ms(lambda: gpu_solve_batch(u0_1, nu_1, [A0], [S0], T, Nt, Nx, Q=Q,
                                          device=DEV, dtype=DT), reps=rep1)
    nl1 = med_ms(lambda: gpu_solve_nl_batch(u0_1, nu_1, [A0], [S0], T, Nt, Nx, beta=3.0,
                                            Q=Q, device=DEV, dtype=DT), reps=rep1)
    Bb = 64 if Nx <= 64 else 8
    u0_b, nu_b = rand_inputs(Bb, Nx)
    ab, sb = [A0]*Bb, [S0]*Bb
    linB = med_ms(lambda: gpu_solve_batch(u0_b, nu_b, ab, sb, T, Nt, Nx, Q=Q,
                                          device=DEV, dtype=DT), reps=2) / Bb
    nlB = med_ms(lambda: gpu_solve_nl_batch(u0_b, nu_b, ab, sb, T, Nt, Nx, beta=3.0,
                                            Q=Q, device=DEV, dtype=DT), reps=2) / Bb
    solver_t[Nx] = (lin1, nl1, linB, nlB)
    log('  %3d^2    %5d  %9.2f  %12.2f  %14.3f  %16.3f'
        % (Nx+1, Nt, lin1, nl1, linB, nlB))

# ================= (B) FrFNO online =================
log('\n================ (B) FrFNO online inference (GPU, ms) ================')
log('Resolution  PT build (offline, s)  B=1 online  B=64/field  speedup vs nonlinear B=1')
frfno_t = {}
pt_build = {}
for Nx, Nt in [(16, 400), (64, 1600), (128, 3200)]:
    AD, SD, PT, tbuild = get_PT(Nx, Nt)
    pt_build[Nx] = tbuild
    PTt = (AD, SD, PT)
    rep = {16: 10, 64: 6, 128: 3}[Nx]
    f1 = med_ms(lambda: frfno_online(1, Nx, PTt), reps=rep)
    Bb = 64 if Nx <= 64 else 16
    fB = med_ms(lambda: frfno_online(Bb, Nx, PTt), reps=3) / Bb
    sp = solver_t[Nx][1] / f1
    frfno_t[Nx] = (f1, fB)
    log('  %3d^2         %8.1f            %8.3f   %9.4f   %8.1fx'
        % (Nx+1, tbuild, f1, fB, sp))

# ================= (C) offline cost & break-even =================
# Offline: training-data generation + network training (measured in b1_burgers runs)
C_DATA_S = 22.0       # 10368 nonlinear reference fields on GPU
C_TRAIN_S = 259.6     # 8000 Adam steps at 17^2
C_PT_S = sum(pt_build.values())
C_OFF = C_DATA_S + C_TRAIN_S
log('\n================ (C) Offline one-time cost & break-even ================')
log('Training-data generation (10368 fields): %.1f s' % C_DATA_S)
log('FrFNO training (8000 steps at 17^2):     %.1f s' % C_TRAIN_S)
log('Propagator-table build (all resolutions, cached after first run): %.1f s' % C_PT_S)
log('Total offline cost: %.1f s' % C_OFF)
log('')
log('Resolution  nonlinear B=1 (ms)  FrFNO B=1 (ms)  net saving/query (ms)  break-even N*')
breakeven = {}
for Nx in [16, 64, 128]:
    nl1 = solver_t[Nx][1]; f1 = frfno_t[Nx][0]; save = nl1 - f1
    nstar = C_OFF / (save / 1000) if save > 0 else float('inf')
    breakeven[Nx] = nstar
    log('  %3d^2      %14.2f   %13.3f   %18.3f   %8.0f queries'
        % (Nx+1, nl1, f1, save, nstar))

# ================= (D) multi-query total wall-clock =================
log('\n================ (D) Total wall-clock vs number of queries (B=1 path) ================')
QLIST = [1, 10, 50, 100, 500, 1000, 5000]
for Nx in [16, 64, 128]:
    nl1_s = solver_t[Nx][1] / 1000
    f1_s = frfno_t[Nx][0] / 1000
    log('')
    log('--- %d^2 : nonlinear solver %.3f s/query, FrFNO %.4f s/query, offline %.1f s ---'
        % (Nx+1, nl1_s, f1_s, C_OFF))
    log('  Queries   solver total (s)   FrFNO total (s)   speedup')
    for nq in QLIST:
        ts = nq * nl1_s
        tf = C_OFF + nq * f1_s
        sp = ts / tf
        log('  %6d    %14.2f    %14.2f   %8.1fx' % (nq, ts, tf, sp))

# ================= summary =================
log('\n================ SUMMARY ================')
log('Per-query speedup (nonlinear solver B=1 / FrFNO B=1):')
for Nx in [16, 64, 128]:
    log('  %d^2: %.1fx' % (Nx+1, solver_t[Nx][1]/frfno_t[Nx][0]))
log('Break-even queries: ' + ', '.join('%d^2 ~ %d' % (Nx+1, breakeven[Nx]) for Nx in [16,64,128]))
log('Done.')
