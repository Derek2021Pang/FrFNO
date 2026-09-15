# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
Rerun ONLY FNO, PINO, CNO for B1, B2-longT, B4-integer.
FNO/PINO: field_dim=2 (was 0)
CNO: official AdamW(1e-3,wd=1e-8) + L1 (was Adam+MSE)
Other models (FrFNO/PDNO/DeepONet/UNet) are NOT retrained.
"""
import os, sys, time
import numpy as np
import torch
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
import frfno_core as FC
import d_scan as DS
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
from frfno_core import (DualNet, build_prop_table, A_GRID, S_GRID,
                           generate_multiscale_initial, DEV, DT)
from gpu_solver import spectral_eigvals_torch
from nl_solver import gpu_solve_nl_batch
from models_surrogate import CNOWrap, n_params
import surrogate_compare as SC

OUT = os.path.join(ROOT, 'rerun_3models_result.txt')
L = []
def log(s): print(s, flush=True); L.append(str(s))
def save(): open(OUT, 'w', encoding='utf-8').write('\n'.join(L))


def run_case(case_name, script_module, T_val, beta, NL_NT, NTR, STEPS, NTEST, RES, MR=32, LAM=0.5):
    """Run FNO/PINO/CNO only for one case."""
    log("\n" + "="*70)
    log("CASE: %s (T=%g, beta=%g, NTR=%d, STEPS=%d, NTEST=%d)" % (case_name, T_val, beta, NTR, STEPS, NTEST))
    log("="*70)

    # import case-specific functions and set order grids
    if case_name == 'B1':
        from b1_burgers import build_trainset_nl, true_field_nl, train_pino
        FC.T = T_val; SC.T = T_val
    elif case_name == 'B2':
        from b2b_longT import build_trainset_nl, true_field_nl, train_pino
        FC.T = T_val; SC.T = T_val
    elif case_name == 'B4':
        from b4_integer import build_trainset_nl, true_field_nl, train_pino, build_pt_b4, train_cno_fair
        FC.T = T_val; SC.T = T_val
        # B4 extends order grids to integer limit
        global A_GRID, S_GRID
        A_GRID = np.linspace(0.6, 1.0, 9).astype(np.float32)
        S_GRID = np.linspace(0.4, 1.0, 9).astype(np.float32)
        SC.A_GRID, SC.S_GRID = A_GRID, S_GRID

    for Nx, _ in RES:
        Phi, e = kl_basis(Nx, KMAX); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Phi, e

    # build training data
    t0 = time.time()
    U0tr, SOL, XItrain, HIST = build_trainset_nl()
    log('training data: %.0fs' % (time.time()-t0))

    SC.MEAN = float(SOL.mean()); SC.SV = float(SOL.std()); SC.NTR = NTR; SC.STEPS = STEPS

    # build propagator tables
    if case_name == 'B4':
        PT = {Nx: build_pt_b4(Nx, Nt) for Nx, Nt in RES}
    else:
        PT = {Nx: build_prop_table(Nx, Nt=NL_NT) for Nx, _ in RES}

    Qmr = spectral_eigvals_torch(MR, DEV, DT)
    log('MEAN=%.4f SV=%.4f' % (SC.MEAN, SC.SV))

    nets = {}; tt = {}

    # --- FNO (field_dim=2) ---
    log('--- Training FNO (field_dim=2) ---')
    t0 = time.time()
    nets['FNO'] = DualNet(2, field_dim=2, seed=2).to(DEV).float()
    tt['FNO'] = SC.train_spectral(nets['FNO'], 'F', SOL, U0tr, XItrain, PT[RES[0][0]], 'FNO')

    # --- PINO (field_dim=2) ---
    log('--- Training PINO (field_dim=2) ---')
    nets['PINO'] = DualNet(2, field_dim=2, seed=7).to(DEV).float()
    tt['PINO'] = train_pino(nets['PINO'], SOL, U0tr, XItrain, HIST, Qmr, 'PINO')

    # --- CNO (official AdamW+L1) ---
    log('--- Training CNO (AdamW+L1) ---')
    nets['CNO'] = CNOWrap(17, ch=32, use_bn=False, seed=3).to(DEV).float()
    if case_name == 'B4':
        tt['CNO'] = train_cno_fair(nets['CNO'], SOL, U0tr, XItrain, 'CNO')
    else:
        tt['CNO'] = SC.train_spatial(nets['CNO'], SOL, U0tr, XItrain, 'CNO',
                                       optim_type='adamw', loss_type='l1')

    # save weights
    wfile = os.path.join(ROOT, '%s_3models_weights.pt' % case_name.lower())
    torch.save({k: nets[k].state_dict() for k in nets}, wfile)
    log('weights saved to %s' % wfile)

    # --- test ---
    rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=NTEST)
    a_te = rng.uniform(A_GRID.min(), A_GRID.max(), NTEST).astype(np.float32)
    s_te = rng.uniform(S_GRID.min(), S_GRID.max(), NTEST).astype(np.float32)
    xi_te = rng.randn(NTEST, KMAX).astype(np.float32)

    rows = {}
    for Nx, Nt in RES:
        Phi, e = PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')]
        Ftr, U0, NU = true_field_nl(Nx, Nt, seeds, a_te, s_te, xi_te, Phi, e)
        tag = '%d^2%s' % (Nx+1, '(train)' if Nx == RES[0][0] else '(zero-shot)')
        log('\n--- resolution %s ---' % tag)
        for name in ['FNO', 'PINO', 'CNO']:
            if name in ('FNO', 'PINO'):
                em, emd, tc = SC.eval_spectral(nets[name], 'F', Nx, U0, NU, a_te, s_te, Ftr, PT[Nx])
            else:
                em, emd, tc = SC.eval_spatial(nets[name], Nx, U0, NU, a_te, s_te, Ftr)
            rows[(Nx, name)] = em
            log('  %-6s rel L2 mean %7.3f%%  (train %.1fs)' % (name, em, tt[name]))

    log('\n--- %s SUMMARY ---' % case_name)
    log('resolution      FNO      PINO      CNO')
    for Nx, _ in RES:
        tag = '%d^2' % (Nx+1)
        log('%-12s %8.3f%% %8.3f%% %8.3f%%' % (tag, rows[(Nx,'FNO')], rows[(Nx,'PINO')], rows[(Nx,'CNO')]))
    log('train time     %6.1fs  %6.1fs  %6.1fs' % (tt['FNO'], tt['PINO'], tt['CNO']))
    save()
    return rows, tt


def main():
    t_total = time.time()

    # B1: T=0.015, beta=3, Nt=400
    run_case('B1', 'b1_burgers', 0.015, 3.0, 400,
             NTR=128, STEPS=8000, NTEST=48,
             RES=[(16, 400), (64, 800), (128, 1600)])

    # B2-longT: T=0.06, beta=3, Nt=1600 (train), test 1600/3200/6400
    run_case('B2', 'b2b_longT', 0.06, 3.0, 1600,
             NTR=128, STEPS=8000, NTEST=48,
             RES=[(16, 1600), (64, 3200), (128, 6400)])

    # B4-integer: T=0.015, beta=3, integer order (alpha=1, s=1)
    run_case('B4', 'b4_integer', 0.015, 3.0, 400,
             NTR=128, STEPS=8000, NTEST=48,
             RES=[(16, 400), (64, 800), (128, 1600)])

    log('\n' + "="*70)
    log('ALL 3 CASES DONE in %.1f min' % ((time.time()-t_total)/60))
    log("="*70)
    save()

if __name__ == '__main__':
    main()
