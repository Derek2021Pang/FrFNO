# -*- coding: utf-8 -*-
"""K scan at Nx_tr=64 (65^2 training grid): fix Nx_tr=64, vary K=4/6/8/12.
Train FrFNO+FNO for each K, save kscan_N64_K{K}.pt, skip if exists (resumable).
Usage: python kscan_N64_train.py [STEPS]
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_SEC = os.path.dirname(_HERE); _PKG_ROOT = os.path.dirname(_SEC)
for _p in (_HERE, _SEC, _PKG_ROOT):
    if _p not in sys.path: sys.path.insert(0, _p)

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
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, 'kscan_N64_train.log')
L = []
def log(s):
    print(s, flush=True); L.append(str(s))
    open(LOG, 'w', encoding='utf-8').write('\n'.join(L))

TN, BETA, NL_NT = 0.015, 3.0, 400
NTR = 128
NX_TR = 64
PAD = 4  # same pad as existing trNx64_K10.pt for fair comparison
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
K_LIST = [4, 6, 8, 12]

def build_trainset():
    Phi, e = kl_basis(NX_TR, KMAX)
    PHI_CACHE[(NX_TR, 'Phi')], PHI_CACHE[(NX_TR, 'e')] = Phi, e
    rng = np.random.RandomState(0)
    U0 = np.stack([generate_multiscale_initial(NX_TR, 8, 1000+i) for i in range(NTR)]).astype(np.float32)
    XI = rng.randn(81, NTR, KMAX).astype(np.float32).reshape(9, 9, NTR, KMAX)
    AA = np.repeat(A_GRID, 9*NTR); SS = np.tile(np.repeat(S_GRID, NTR), 9)
    NU = make_nu(NX_TR, XI.reshape(81*NTR, KMAX), Phi, e)
    UB = np.concatenate([U0]*81, 0); Q = spectral_eigvals_torch(NX_TR, DEV, DT)
    sols = []; t0 = time.time()
    for i0 in range(0, len(AA), 512):
        fo = gpu_solve_nl_batch(UB[i0:i0+512], NU[i0:i0+512], AA[i0:i0+512], SS[i0:i0+512],
                                TN, NL_NT, NX_TR, beta=BETA, Q=Q, device=DEV, dtype=DT)
        sols.append(fo.cpu().numpy())
    SOL = np.concatenate(sols, 0).reshape(9, 9, NTR, NX_TR+1, NX_TR+1)
    log(f'Nx_tr={NX_TR}: nonlinear training ground truth, {9*9*NTR} cases, {time.time()-t0:.0f}s')
    return U0, SOL, XI

def train_K(net, kind, K, SOL, U0, XI, PT, MEAN, SV, tag):
    import torch.optim as opt
    op = opt.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = opt.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5); net.train(); t0 = time.time()
    rng = np.random.RandomState(123)
    Phi, e = PHI_CACHE[(NX_TR, 'Phi')], PHI_CACHE[(NX_TR, 'e')]
    for st in range(STEPS+1):
        ia, js = rng.randint(9), rng.randint(9)
        itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
        a, s = A_GRID[ia], S_GRID[js]
        sol = torch.tensor(SOL[ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0[itr], device=DEV, dtype=DT)
        nub = torch.tensor(make_nu(NX_TR, XI[ia, js, itr], Phi, e), device=DEV, dtype=DT)
        if kind == 'A':
            gs = [prop_at(a, s, *PT)]+[prop_at(*z, *PT) for z in [(a-SC.DA, s), (a+SC.DA, s), (a, s-SC.DSNB), (a, s+SC.DSNB)]]
            ubs = [ub_full_batch(U0[itr], g, NX_TR) for g in gs]
            xb = torch.stack([u0b, ubs[0], nub, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            tg = (sol-ubs[0])/SV
        else:
            xb = torch.stack([u0b, nub], -1); tg = (sol-MEAN)/SV
        loss = F.mse_loss(net(xb, (a, s)), tg)
        op.zero_grad(); loss.backward(); op.step(); sch.step()
        if st % 1000 == 0: log(f'  [{tag}] step {st} loss {float(loss):.5e}')
    net.eval(); dt = time.time()-t0
    log(f'  [{tag}] train {dt:.0f}s params {n_params(net)/1e3:.1f}K')
    return dt

def main():
    LG.T = TN; SC.T = TN; SC.STEPS = STEPS; SC.NTR = NTR
    all_done = all(os.path.exists(os.path.join(OUT, f'kscan_N64_K{K}.pt')) for K in K_LIST)
    if all_done:
        log('All K models already exist, nothing to do.'); return

    # Build/cache training set once
    cache_path = os.path.join(OUT, f'trainset_N64.npz')
    if os.path.exists(cache_path):
        log('Loading cached training set...')
        d = np.load(cache_path, allow_pickle=True)
        U0, SOL, XI = d['U0'], d['SOL'], d['XI']
        MEAN, SV = float(d['MEAN']), float(d['SV'])
    else:
        U0, SOL, XI = build_trainset()
        MEAN, SV = float(SOL.mean()), float(SOL.std())
        np.savez(cache_path, U0=U0, SOL=SOL, XI=XI, MEAN=MEAN, SV=SV)

    log(f'MEAN={MEAN:.4f} SV={SV:.4f} STEPS={STEPS}')
    PT = build_prop_table(NX_TR, Nt=NL_NT)

    for K in K_LIST:
        wpath = os.path.join(OUT, f'kscan_N64_K{K}.pt')
        if os.path.exists(wpath):
            log(f'K={K} already exists, skip'); continue
        tK = time.time()
        ck = {'Nx_tr': NX_TR, 'K': K, 'pad': PAD, 'MEAN': MEAN, 'SV': SV}
        for name, kind, seed, fd in [('FrFNO', 'A', 1, 1), ('FNO', 'F', 2, 0)]:
            net = DualNet(7 if kind == 'A' else 2, modes=K, pad=PAD, field_dim=fd, seed=seed).to(DEV).float()
            tt = train_K(net, kind, K, SOL, U0, XI, PT, MEAN, SV, f'K{K}-{name}')
            ck[name] = net.state_dict(); ck[f'{name}_t'] = tt; ck[f'{name}_np'] = n_params(net)/1e3
            torch.save(ck, wpath)
            log(f'K={K} {name} done and saved')
        log(f'K={K} done in {time.time()-tK:.0f}s')
    log('ALL K TRAIN DONE')

if __name__ == '__main__':
    main()
