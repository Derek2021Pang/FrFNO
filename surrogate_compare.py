# -*- coding: utf-8 -*-
"""
Fair comparison of FrFNO as a forward surrogate for fractional PDEs against mainstream neural operators (surrogate models).
Task: (u0, nu field, alpha, s) -> u(T); one network covers all random orders / coefficient fields.
Models: FrFNO (analytic propagator + dynamic fractional wavenumber + residual) / standard FNO (same spectral backbone, no physics prior) /
      CNO (off-the-shelf CNO2d) / DeepONet / UNet.
Fairness: identical training data (the same per-batch random sequence), equal step counts / optimizer / batch, parameter counts of the same order (1.1-2.4M, reported truthfully).
Evaluation: held-out accuracy at the same 17^2 resolution; zero-shot cross-resolution 17->65->129 without retraining; training / inference time.
"""
import os, sys, time
os.environ.setdefault('FAST', '0')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import d_scan as DS
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
from frfno_core import (DualNet, build_prop_table, prop_at, ub_full_batch,
                           generate_multiscale_initial, A_GRID, S_GRID, T, DEV, DT)
from gpu_solver import gpu_solve_imex_batch, spectral_eigvals_torch
from models_surrogate import CNOWrap, DeepONet2d, UNet2d, n_params
DA = 0.10; DSNB = 0.08          # neighbor-order stride (consistent with the existing FrFNO)

FAST = os.environ.get('FAST') == '1'
NTR = 16 if FAST else 128
STEPS = 30 if FAST else 8000
NTEST = 8 if FAST else 48
RES = [(16, 400), (64, 800), (128, 1600)]     # (Nx,Nt) train at 17 / super-resolution 65 / 129
OUT = os.path.join(ROOT, 'surrogate_compare_result.txt')
L = []
def log(s): print(s, flush=True); L.append(str(s))
def save(): open(OUT, 'w', encoding='utf-8').write('\n'.join(L))
def rel(a, b): return np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12)

def train_spectral(net, kind, SOL, U0, XI, PT, tag):
    import torch.optim as opt
    op = opt.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = opt.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5); net.train(); t0 = time.time()
    rng = np.random.RandomState(123); Phi, e = PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')]
    for st in range(STEPS+1):
        ia, js = rng.randint(9), rng.randint(9); itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
        a, s = A_GRID[ia], S_GRID[js]
        sol = torch.tensor(SOL[ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0[itr], device=DEV, dtype=DT)
        nub = torch.tensor(make_nu(16, XI[ia, js, itr], Phi, e), device=DEV, dtype=DT)
        if kind == 'A':
            gs = [prop_at(a, s, *PT)]+[prop_at(*z, *PT) for z in
                  [(a-DA, s), (a+DA, s), (a, s-DSNB), (a, s+DSNB)]]
            ubs = [ub_full_batch(U0[itr], g, 16) for g in gs]
            xb = torch.stack([u0b, ubs[0], nub, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            tg = (sol-ubs[0])/SV
        else:
            xb = torch.stack([u0b, nub], -1); tg = (sol-MEAN)/SV
        loss = F.mse_loss(net(xb, (a, s)), tg)
        op.zero_grad(); loss.backward(); op.step(); sch.step()
    net.eval(); log('  [%-7s] train %ds  params %.1fK' % (tag, time.time()-t0, n_params(net)/1e3)); return time.time()-t0

def train_spatial(net, SOL, U0, XI, tag, optim_type='adam', loss_type='mse'):
    import torch.optim as opt
    if optim_type == 'adamw':
        op = opt.AdamW(net.parameters(), 1e-3, weight_decay=1e-8)
    else:
        op = opt.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = opt.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5); net.train(); t0 = time.time()
    rng = np.random.RandomState(123); Phi, e = PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')]
    for st in range(STEPS+1):
        ia, js = rng.randint(9), rng.randint(9); itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
        a, s = A_GRID[ia], S_GRID[js]
        sol = torch.tensor(SOL[ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0[itr], device=DEV, dtype=DT)
        nub = torch.tensor(make_nu(16, XI[ia, js, itr], Phi, e), device=DEV, dtype=DT)
        x = torch.stack([u0b, nub], 1)
        cond = torch.tensor(np.stack([np.full(len(itr), a), np.full(len(itr), s)], 1), device=DEV, dtype=DT)
        o = net(x, cond).unsqueeze(1)
        tg = ((sol-MEAN)/SV).unsqueeze(1)
        if loss_type == 'l1':
            loss = F.l1_loss(o, tg)
        else:
            loss = F.mse_loss(o, tg)
        op.zero_grad(); loss.backward(); op.step(); sch.step()
    net.eval(); log('  [%-7s] train %ds  params %.1fK' % (tag, time.time()-t0, n_params(net)/1e3)); return time.time()-t0

def true_field(Nx, Nt, seeds, a, s, xi, Phi, e):
    U0 = np.stack([generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
    NU = make_nu(Nx, xi, Phi, e)
    Q = spectral_eigvals_torch(Nx, DEV, DT)
    F = gpu_solve_imex_batch(U0, NU, a, s, T, Nt, Nx, Q=Q, device=DEV, dtype=DT).cpu().numpy()
    return F.reshape(len(a), -1), U0, NU

@torch.no_grad()
def eval_spectral(net, kind, Nx, U0, NU, a, s, Ftrue, PT):
    t0 = time.time(); errs = []
    for i in range(len(a)):
        u0t = torch.tensor(U0[i:i+1], device=DEV, dtype=DT)
        nut = torch.tensor(NU[i:i+1], device=DEV, dtype=DT); aa, ss = float(a[i]), float(s[i])
        if kind == 'A':
            gs = [prop_at(aa, ss, *PT)]+[prop_at(*z, *PT) for z in
                  [(aa-DA, ss), (aa+DA, ss), (aa, ss-DSNB), (aa, ss+DSNB)]]
            ubs = [ub_full_batch(U0[i:i+1], g, Nx) for g in gs]
            xb = torch.stack([u0t, ubs[0], nut, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            phys = ubs[0].cpu().numpy()+SV*net(xb, (aa, ss)).cpu().numpy()
        else:
            xb = torch.stack([u0t, nut], -1)
            phys = MEAN+SV*net(xb, (aa, ss)).cpu().numpy()
        errs.append(rel(phys.reshape(-1), Ftrue[i]))
    return 100*np.mean(errs), 100*np.median(errs), time.time()-t0

@torch.no_grad()
def eval_spatial(net, Nx, U0, NU, a, s, Ftrue):
    t0 = time.time()
    x = torch.tensor(np.stack([U0, NU], 1), device=DEV, dtype=DT)
    cond = torch.tensor(np.stack([a, s], 1), device=DEV, dtype=DT)
    o = net(x, cond).cpu().numpy().reshape(len(a), -1)
    phys = MEAN+SV*o
    errs = [rel(phys[i], Ftrue[i]) for i in range(len(a))]
    return 100*np.mean(errs), 100*np.median(errs), time.time()-t0

def main():
    global MEAN, SV
    for Nx, _ in RES:
        Phi, e = kl_basis(Nx, KMAX, 1.0); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Phi, e
    U0tr, SOL, XItrain = DS.build_trainset(); MEAN = float(SOL.mean()); SV = float(SOL.std())
    PT = {Nx: build_prop_table(Nx, Nt=400) for Nx, _ in RES}
    log('surrogate comparison: NTR=%d STEPS=%d Ntest=%d; MEAN=%.4f SV=%.4f' % (NTR, STEPS, NTEST, MEAN, SV))
    nets = {}
    tt = {}
    nets['FrFNO'] = DualNet(7, seed=1); tt['FrFNO'] = train_spectral(nets['FrFNO'], 'A', SOL, U0tr, XItrain, PT[16], 'FrFNO')
    nets['FNO'] = DualNet(2, field_dim=0, seed=2); tt['FNO'] = train_spectral(nets['FNO'], 'F', SOL, U0tr, XItrain, PT[16], 'FNO')
    nets['CNO'] = CNOWrap(17, ch=32, seed=3).to(DEV).float(); tt['CNO'] = train_spatial(nets['CNO'], SOL, U0tr, XItrain, 'CNO')
    nets['DeepONet'] = DeepONet2d(w=384, seed=4).to(DEV).float(); tt['DeepONet'] = train_spatial(nets['DeepONet'], SOL, U0tr, XItrain, 'DeepONet')
    nets['UNet'] = UNet2d(base=48, seed=5).to(DEV).float(); tt['UNet'] = train_spatial(nets['UNet'], SOL, U0tr, XItrain, 'UNet')
    torch.save({k: nets[k].state_dict() for k in nets}, os.path.join(ROOT, 'surrogate_weights.pt'))
    # test conditions (continuous randomness, unlike the 9x9 discrete training grid)
    rng = np.random.RandomState(2024)
    seeds = rng.randint(100000, size=NTEST)
    a_te = rng.uniform(A_GRID.min(), A_GRID.max(), NTEST).astype(np.float32)
    s_te = rng.uniform(S_GRID.min(), S_GRID.max(), NTEST).astype(np.float32)
    xi_te = rng.randn(NTEST, KMAX).astype(np.float32)
    rows = {}
    for Nx, Nt in RES:
        Phi, e = PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')]
        Ftr, U0, NU = true_field(Nx, Nt, seeds, a_te, s_te, xi_te, Phi, e)
        log('\n===== resolution %d^2 (trained at 17^2, %s) =====' % (Nx+1, 'same resolution' if Nx == 16 else 'zero-shotsuper-resolution'))
        for name in ['FrFNO', 'FNO', 'CNO', 'DeepONet', 'UNet']:
            if name in ('FrFNO', 'FNO'):
                em, emd, tc = eval_spectral(nets[name], 'A' if name == 'FrFNO' else 'F',
                                            Nx, U0, NU, a_te, s_te, Ftr, PT[Nx])
            else:
                em, emd, tc = eval_spatial(nets[name], Nx, U0, NU, a_te, s_te, Ftr)
            rows[(Nx, name)] = (em, emd, tc)
            log('  %-8s relative L2 mean%7.3f%%  median%7.3f%%  inference%.3fs' % (name, em, emd, tc))
        save()
    log('\n================ summary: relative L2 mean error (%) ================')
    hdr = 'resolution   ' + ''.join('%10s' % m for m in ['FrFNO', 'FNO', 'CNO', 'DeepONet', 'UNet'])
    log(hdr)
    for Nx, _ in RES:
        tag = '%d^2%s' % (Nx+1, '(train)' if Nx == 16 else '(zero-shot)')
        log('%-9s' % tag + ''.join('%9.3f%%' % rows[(Nx, m)][0] for m in ['FrFNO', 'FNO', 'CNO', 'DeepONet', 'UNet']))
    log('training time ' + ''.join('%9.1fs' % tt[m] for m in ['FrFNO', 'FNO', 'CNO', 'DeepONet', 'UNet']))
    log('parameter count/K ' + ''.join('%9.1f' % (n_params(nets[m])/1e3) for m in ['FrFNO', 'FNO', 'CNO', 'DeepONet', 'UNet']))
    np.savez(os.path.join(ROOT, 'surrogate_compare_metrics.npz'),
             rows=np.array([[rows[(Nx, m)] for m in ['FrFNO', 'FNO', 'CNO', 'DeepONet', 'UNet']] for Nx, _ in RES]),
             methods=np.array(['FrFNO', 'FNO', 'CNO', 'DeepONet', 'UNet']),
             res=np.array([Nx+1 for Nx, _ in RES]), train_t=np.array([tt[m] for m in ['FrFNO', 'FNO', 'CNO', 'DeepONet', 'UNet']]))
    save()

if __name__ == '__main__':
    main()
