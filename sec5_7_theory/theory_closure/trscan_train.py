# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SEC = _os.path.dirname(_HERE); _PKG_ROOT = _os.path.dirname(_SEC)
for _p in (_HERE, _SEC, _PKG_ROOT):
    if _p not in _sys.path: _sys.path.insert(0, _p)
"""P0-1 supplementary axis: training (physical) resolution scan, fixing spectral modes K=10/pad=4 and varying only the training-label grid Nx_tr=16/32/64
(17/33/65 points). surrogate_compare.train_spectral hard-codes the training grid to 16, so its training loop is copied here and
parameterized by Nx_tr. Train FrFNO+FNO at each resolution, save trNx{Nx}_K10.pt (with its own MEAN/SV), and skip if present.
Usage: python trscan_train.py <Nx_tr> [STEPS]"""
import os, sys, time
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
import numpy as np, torch, torch.nn.functional as F
import frfno_legacy as LG
from frfno_legacy import (DualNet, build_prop_table, A_GRID, S_GRID, DEV, DT,
                           generate_multiscale_initial, prop_at, ub_full_batch)
from gpu_solver import spectral_eigvals_torch
from nl_solver import gpu_solve_nl_batch
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
import surrogate_compare as SC
from models_surrogate import n_params

OUT = os.path.join(ROOT, 'theory_closure')
TN, BETA, NL_NT, K, PAD = 0.015, 3.0, 400, 10, 4
NTR = 128; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
Nx_tr = int(sys.argv[1])
WPATH = os.path.join(OUT, f'trNx{Nx_tr}_K{K}.pt')
L = []
def log(s): print(s, flush=True); L.append(str(s)); open(os.path.join(OUT, f'trscan_{Nx_tr}.log'), 'w', encoding='utf-8').write('\n'.join(L))

def build_trainset(Nx_tr):
    Phi, e = kl_basis(Nx_tr, KMAX); PHI_CACHE[(Nx_tr, 'Phi')], PHI_CACHE[(Nx_tr, 'e')] = Phi, e
    rng = np.random.RandomState(0)
    U0 = np.stack([generate_multiscale_initial(Nx_tr, 8, 1000+i) for i in range(NTR)]).astype(np.float32)
    XI = rng.randn(81, NTR, KMAX).astype(np.float32).reshape(9, 9, NTR, KMAX)
    AA = np.repeat(A_GRID, 9*NTR); SS = np.tile(np.repeat(S_GRID, NTR), 9)
    NU = make_nu(Nx_tr, XI.reshape(81*NTR, KMAX), Phi, e)
    UB = np.concatenate([U0]*81, 0); Q = spectral_eigvals_torch(Nx_tr, DEV, DT)
    sols = []; t0 = time.time()
    for i0 in range(0, len(AA), 512):
        fo = gpu_solve_nl_batch(UB[i0:i0+512], NU[i0:i0+512], AA[i0:i0+512], SS[i0:i0+512],
                                TN, NL_NT, Nx_tr, beta=BETA, Q=Q, device=DEV, dtype=DT)
        sols.append(fo.cpu().numpy())
    SOL = np.concatenate(sols, 0).reshape(9, 9, NTR, Nx_tr+1, Nx_tr+1)
    log(f'Nx_tr={Nx_tr}: nonlinear training ground truth, {9*9*NTR} cases, {time.time()-t0:.0f}s')
    return U0, SOL, XI

def train_Nx(net, kind, Nx_tr, SOL, U0, XI, PT, MEAN, SV, tag):
    import torch.optim as opt
    op = opt.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = opt.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5); net.train(); t0 = time.time()
    rng = np.random.RandomState(123); Phi, e = PHI_CACHE[(Nx_tr, 'Phi')], PHI_CACHE[(Nx_tr, 'e')]
    for st in range(STEPS+1):
        ia, js = rng.randint(9), rng.randint(9); itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
        a, s = A_GRID[ia], S_GRID[js]
        sol = torch.tensor(SOL[ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0[itr], device=DEV, dtype=DT)
        nub = torch.tensor(make_nu(Nx_tr, XI[ia, js, itr], Phi, e), device=DEV, dtype=DT)
        if kind == 'A':
            gs = [prop_at(a, s, *PT)]+[prop_at(*z, *PT) for z in [(a-SC.DA, s), (a+SC.DA, s), (a, s-SC.DSNB), (a, s+SC.DSNB)]]
            ubs = [ub_full_batch(U0[itr], g, Nx_tr) for g in gs]
            xb = torch.stack([u0b, ubs[0], nub, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            tg = (sol-ubs[0])/SV
        else:
            xb = torch.stack([u0b, nub], -1); tg = (sol-MEAN)/SV
        loss = F.mse_loss(net(xb, (a, s)), tg)
        op.zero_grad(); loss.backward(); op.step(); sch.step()
        if st % 1000 == 0: log(f'  [{tag}] step {st} loss {float(loss):.5e}')
    net.eval(); dt = time.time()-t0; log(f'  [{tag}] train {dt:.0f}s params {n_params(net)/1e3:.1f}K'); return dt

def main():
    if os.path.exists(WPATH): log('already exists, skip'); return
    LG.T = TN; SC.T = TN
    U0, SOL, XI = build_trainset(Nx_tr)
    MEAN, SV = float(SOL.mean()), float(SOL.std())
    log(f'MEAN={MEAN:.4f} SV={SV:.4f} STEPS={STEPS}')
    PT = build_prop_table(Nx_tr, Nt=NL_NT)
    ck = {'Nx_tr': Nx_tr, 'K': K, 'pad': PAD, 'MEAN': MEAN, 'SV': SV}
    for name, kind, seed, fd in [('FrFNO', 'A', 1, 1), ('FNO', 'F', 2, 0)]:
        net = DualNet(7 if kind == 'A' else 2, modes=K, pad=PAD, field_dim=fd, seed=seed).to(DEV).float()
        tt = train_Nx(net, kind, Nx_tr, SOL, U0, XI, PT, MEAN, SV, f'{Nx_tr}-{name}')
        ck[name] = net.state_dict(); ck[f'{name}_t'] = tt; ck[f'{name}_np'] = n_params(net)/1e3
        torch.save(ck, WPATH)
        log(f'Nx_tr={Nx_tr} {name} done and saved')
    torch.save(ck, WPATH); log('trscan DONE')

if __name__ == '__main__':
    main()
