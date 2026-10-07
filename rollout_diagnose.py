# -*- coding: utf-8 -*-
"""Cheap diagnostic for why FrFNO's 4-step rollout error grows faster than FNO's.

Three measurements, all reusing the paper's B1 weights (b1_weights.pt), the
unified 1-D propagator table, the Gamma(2-alpha) corrected L1 coefficients,
and the same 48-sample held-out stream (RandomState(2024)).

(A) Per-step error decomposition: record the mean rel-L2 error after each of
    the four autoregressive steps, for FrFNO and FNO, at 17^2 and 129^2.
(B) Input-perturbation sensitivity: feed u0 + delta with delta of the size of
    a typical single-step error. For FrFNO, split the response into the linear
    propagator part and the residual-network part; for FNO, measure the whole
    map. Report the amplification factor
    (||out(u0+delta)-out(u0)||/||out(u0)||) / (||delta||/||u0||).
(C) Error spectrum: radial power spectrum of the step-1 error vs the step-4
    error at 17^2 (low-frequency accumulation check).

This is a supporting diagnostic for the rollout discussion in Sec. 5.2
(not a numbered table in the paper). Run time on a desktop GPU: a few minutes.
"""
import os, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import frfno_core as FC
import d_scan as DS
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
from frfno_core import (DualNet, build_prop_table, prop_at, ub_full_batch,
                        generate_multiscale_initial, A_GRID, S_GRID, DEV, DT)
from gpu_solver import spectral_eigvals_torch
from nl_solver import gpu_solve_nl_batch
import surrogate_compare as SC
import b1_burgers as B1

TN, BETA = 0.015, 3.0
T_LONG = 0.06
STEPS_AR = 4
NTEST = 48
NTEST_SMALL = 12                    # 129^2 truth is expensive, use 12 samples
OUT = os.path.join(ROOT, 'rollout_diagnose_result.txt')
L = []
def log(s):
    print(s, flush=True); L.append(str(s)); open(OUT, 'w', encoding='utf-8').write('\n'.join(L))

def rel(a, b): return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)

def step_spectral(net, kind, Nx, U0b, NU, a, s, PT):
    """One single-step prediction. U0b/NU: (1,H,W) tensors. Returns (H,W) ndarray
    plus (linear_part, residual_part) for FrFNO (None for FNO)."""
    aa, ss = float(a), float(s)
    if kind == 'A':
        gs = [prop_at(aa, ss, *PT)] + [prop_at(*z, *PT) for z in
              [(aa - SC.DA, ss), (aa + SC.DA, ss), (aa, ss - SC.DSNB), (aa, ss + SC.DSNB)]]
        ubs = [ub_full_batch(U0b, g, Nx) for g in gs]
        xb = torch.stack([U0b, ubs[0], NU, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
        lin = ubs[0].cpu().numpy()
        res = SC.SV * net(xb, (aa, ss)).cpu().numpy()
        return lin + res, lin, res
    else:
        xb = torch.stack([U0b, NU], -1)
        phys = SC.MEAN + SC.SV * net(xb, (aa, ss)).cpu().numpy()
        return phys, None, None

def radial_spec(err, Nx):
    """Radially averaged power spectrum of a 2-D error field (float64)."""
    F = np.fft.rfft2(err - err.mean())
    P = np.abs(F) ** 2
    kx = np.fft.fftfreq(Nx + 1)[: P.shape[1]]
    ky = np.fft.fftfreq(Nx + 1)[: P.shape[0]]
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    kb = np.linspace(0, K.max(), 20)
    spec = []
    for i in range(len(kb) - 1):
        m = (K >= kb[i]) & (K < kb[i + 1])
        spec.append(P[m].mean() if m.any() else 0.0)
    return np.array(spec), kb[:-1]

def main():
    FC.T = TN; SC.T = TN; B1.FC.T = TN
    for Nx, _ in [(16, 400), (128, 6400)]:
        Phi, e = kl_basis(Nx, KMAX, 1.0); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Phi, e
    log('rebuilding B1 training ground truth for MEAN/SV (one-time, ~minutes)...')
    t0 = time.time(); U0tr, SOL, XItrain, HIST = B1.build_trainset_nl()
    SC.MEAN = float(SOL.mean()); SC.SV = float(SOL.std())
    log('MEAN=%.4f SV=%.4f (%.0fs)' % (SC.MEAN, SC.SV, time.time() - t0))
    fr = DualNet(7, seed=1); fno = DualNet(2, field_dim=2, seed=2)
    sd = torch.load(os.path.join(ROOT, 'b1_weights.pt'), map_location=DEV)
    fr.load_state_dict(sd['FrFNO']); fr.eval()
    fno.load_state_dict(sd['FNO']); fno.eval()
    log('loaded b1_weights.pt (FrFNO, FNO)')
    rng = np.random.RandomState(2024)
    seeds = rng.randint(100000, size=NTEST)
    a_te = rng.uniform(A_GRID.min(), A_GRID.max(), NTEST).astype(np.float32)
    s_te = rng.uniform(S_GRID.min(), S_GRID.max(), NTEST).astype(np.float32)
    xi_te = rng.randn(NTEST, KMAX).astype(np.float32)
    rng2 = np.random.RandomState(777)       # fresh stream for the gaussian deltas
    with torch.no_grad():
        for Nx, Nt in [(16, 400), (128, 6400)]:
            n = NTEST if Nx == 16 else NTEST_SMALL
            Phi, e = PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')]
            U0 = np.stack([generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds[:n]]).astype(np.float32)
            NU = make_nu(Nx, xi_te[:n], Phi, e)
            Q = spectral_eigvals_torch(Nx, DEV, DT)
            tag = '%d^2' % (Nx + 1)
            log('\n===== diagnostic at %s (Nt=%d truth) =====' % (tag, Nt))
            PT = build_prop_table(Nx, Nt=400)      # single-step table, T=0.015
            # T=0.015 reference for the step-1 error field (cheap at 17^2 only)
            Ftr1 = None
            if Nx == 16:
                Ftr1 = gpu_solve_nl_batch(U0, NU, a_te[:n], s_te[:n], TN, 400, Nx,
                                          beta=BETA, Q=Q, device=DEV, dtype=DT).cpu().numpy()
                Ftr1 = Ftr1.reshape(n, -1)
            Ftr4 = gpu_solve_nl_batch(U0, NU, a_te[:n], s_te[:n], T_LONG, Nt, Nx,
                                      beta=BETA, Q=Q, device=DEV, dtype=DT).cpu().numpy()
            Ftr4 = Ftr4.reshape(n, -1)
            # (A) per-step errors: each step compares against the truth at the SAME time
            #     T = s*0.015 (17^2 solves four cheap references; 129^2 uses the
            #     final-T reference already available and its total error from
            #     rollout_autoreg_metrics.npz is not recomputed here).
            if Nx == 16:
                Fref = {}
                for s in range(1, STEPS_AR + 1):
                    Fref[s] = gpu_solve_nl_batch(U0, NU, a_te[:n], s_te[:n], s * TN, s * 400, Nx,
                                                 beta=BETA, Q=Q, device=DEV, dtype=DT).cpu().numpy()
                    Fref[s] = Fref[s].reshape(n, -1)
            for name, net, kind in [('FrFNO', fr, 'A'), ('FNO', fno, 'F')]:
                step_errs = {s: [] for s in range(1, STEPS_AR + 1)}
                for i in range(n):
                    u = U0[i:i + 1].copy()
                    for s in range(1, STEPS_AR + 1):
                        u0t = torch.tensor(u, device=DEV, dtype=DT)
                        nut = torch.tensor(NU[i:i + 1], device=DEV, dtype=DT)
                        p, _, _ = step_spectral(net, kind, Nx, u0t, nut, a_te[i], s_te[i], PT)
                        u = p.reshape(1, Nx + 1, Nx + 1).astype(np.float32)
                        if Nx == 16:
                            step_errs[s].append(rel(u.reshape(-1), Fref[s][i]))
                        else:
                            step_errs[s].append(rel(u.reshape(-1), Ftr4[i]))
                if Nx == 16:
                    line = '  %-6s per-step rel L2 (vs same-time truth):' % name + ''.join(
                        '  step%d %7.3f%%' % (s, 100 * np.mean(step_errs[s])) for s in range(1, 5))
                else:
                    line = '  %-6s per-step rel L2 (vs T=0.06 truth; off-time refs):' % name + ''.join(
                        '  step%d %7.3f%%' % (s, 100 * np.mean(step_errs[s])) for s in range(1, 5))
                log(line)
                if name == 'FrFNO' and Ftr1 is not None:
                    # step-1 error field for the real-delta sensitivity
                    u0t = torch.tensor(U0, device=DEV, dtype=DT)
                    nut = torch.tensor(NU, device=DEV, dtype=DT)
                    p1 = np.stack([step_spectral(net, kind, Nx, u0t[i:i + 1], nut[i:i + 1],
                                                 a_te[i], s_te[i], PT)[0] for i in range(n)])
                    e1 = (p1.reshape(n, -1) - Ftr1)
            # (D) time-shift invariance of the linear propagator (17^2 only).
            #     The single-step table E_alpha(tau^alpha Lambda) is exact from
            #     t=0 (Duhamel kernel for the Caputo history). Applying it to the
            #     field at t=0.015 (i.e. restarting the memory at 0.015) is NOT
            #     the same as evolving 0.015->0.030, because the Caputo memory
            #     starts at t=0. Compare the pure linear table prediction
            #     E(tau)u(0.015) against the true u(0.030).
            if Nx == 16:
                lin_shift = []
                for i in range(n):
                    u1t = torch.tensor(Fref[1][i].reshape(1, Nx + 1, Nx + 1), device=DEV, dtype=DT)
                    g = prop_at(float(a_te[i]), float(s_te[i]), *PT)
                    v = ub_full_batch(u1t, g, Nx).cpu().numpy().reshape(-1)
                    lin_shift.append(rel(v, Fref[2][i]))
                log('  linear-prop shift error E(tau)u(0.015) vs u(0.030):  mean %6.3f%%'
                    % (100 * np.mean(lin_shift)))
            # (B) sensitivity: gaussian delta of single-step-error size
            for scale_name, scale in [('gauss', None)]:
                for name, net, kind, parts in [('FrFNO', fr, 'A', True), ('FNO', fno, 'F', False)]:
                    amps = []
                    for i in range(n):
                        u0 = U0[i:i + 1].copy()
                        u0t = torch.tensor(u0, device=DEV, dtype=DT)
                        nut = torch.tensor(NU[i:i + 1], device=DEV, dtype=DT)
                        y0, lin0, res0 = step_spectral(net, kind, Nx, u0t, nut, a_te[i], s_te[i], PT)
                        nrm = np.linalg.norm(y0) + 1e-12
                        # typical single-step error size: ~2% of the field RMS
                        delta = (0.02 * (np.linalg.norm(u0) / np.sqrt(u0.size))) * rng2.randn(*u0.shape).astype(np.float32)
                        u1 = u0 + delta
                        u1t = torch.tensor(u1, device=DEV, dtype=DT)
                        y1, lin1, res1 = step_spectral(net, kind, Nx, u1t, nut, a_te[i], s_te[i], PT)
                        din = np.linalg.norm(delta) / (np.linalg.norm(u0) + 1e-12)
                        if parts:
                            amp_lin = (np.linalg.norm(lin1 - lin0) / nrm) / (din + 1e-12)
                            amp_res = (np.linalg.norm(res1 - res0) / nrm) / (din + 1e-12)
                            amp_tot = (np.linalg.norm(y1 - y0) / nrm) / (din + 1e-12)
                            amps.append((amp_lin, amp_res, amp_tot))
                        else:
                            amp_tot = (np.linalg.norm(y1 - y0) / nrm) / (din + 1e-12)
                            amps.append((amp_tot,))
                    A = np.array(amps)
                    if parts:
                        log('  %-6s sensitivity (2%%-RMS input delta):  linear %.3f  residual %.3f  total %.3f'
                            % (name, A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean()))
                    else:
                        log('  %-6s sensitivity (2%%-RMS input delta):  total %.3f' % (name, A[:, 0].mean()))
            # (C) error spectrum at 17^2: step1 vs step4 error
            if Nx == 16 and Ftr1 is not None:
                for name, net, kind in [('FrFNO', fr, 'A'), ('FNO', fno, 'F')]:
                    e1s, e4s = [], []
                    for i in range(n):
                        u = U0[i:i + 1].copy()
                        for s in range(1, 5):
                            u0t = torch.tensor(u, device=DEV, dtype=DT)
                            nut = torch.tensor(NU[i:i + 1], device=DEV, dtype=DT)
                            p, _, _ = step_spectral(net, kind, Nx, u0t, nut, a_te[i], s_te[i], PT)
                            u = p.reshape(1, Nx + 1, Nx + 1).astype(np.float32)
                            if s == 1:
                                e1s.append((p.reshape(-1) - Ftr1[i]).reshape(Nx + 1, Nx + 1))
                        e4s.append((u.reshape(-1) - Ftr4[i]).reshape(Nx + 1, Nx + 1))
                    sp1, kb = radial_spec(np.mean(e1s, 0), Nx)
                    sp4, _ = radial_spec(np.mean(e4s, 0), Nx)
                    lo = sp1[: len(sp1) // 2].sum() / (sp1.sum() + 1e-12)
                    lo4 = sp4[: len(sp4) // 2].sum() / (sp4.sum() + 1e-12)
                    log('  %-6s 17^2 error spectrum: low-k share step1 %.2f  step4 %.2f'
                        % (name, lo, lo4))
    log('\n========= diagnostic summary =========')
    log('(A) per-step errors isolate WHICH step explodes.')
    log('(B) sensitivity separates the linear propagator from the residual network for FrFNO.')
    log('(C) low-k share of the error shows whether low-frequency modes accumulate.')
    log('done.')

if __name__ == '__main__':
    main()
