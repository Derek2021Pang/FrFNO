# -*- coding: utf-8 -*-
"""
B1 nonlinearfractional-order Burgers: ^CD_t^a u + beta*u*d_x u = -nu(x,y)(-Delta)^s u, T=0.015, beta=3。
FrFNO propagator uses onlylinearanalytic solution (no advection), tests whether"linearphysics coarse solution + residual network"undernonlinearholds。
7 models: FrFNO / FNO / PINO(FNObackbone + high-resolutionPDE residual) / PDNO(continuoussymbol) / CNO / DeepONet / UNet。
FAST=1 smoke test(NTR16/STEPS30/Ntest8/only17^2); defaultfull。
"""
import os, sys, time
os.environ.setdefault('FAST', '0')
import numpy as np
import torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import frfno_core as FC
import d_scan as DS
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
from frfno_core import (DualNet, build_prop_table, A_GRID, S_GRID,
                           generate_multiscale_initial, DEV, DT)
from gpu_solver import spectral_eigvals_torch, dst2_torch
from nl_solver import gpu_solve_nl_batch, ddx_upwind
from models_surrogate import CNOWrap, DeepONet2d, UNet2d, n_params
from models_extra import PDNO2d
import surrogate_compare as SC

TN, BETA, NL_NT = 0.015, 3.0, 400
MR = 32                                   # PINO PDE residualcollocationresolution(33^2)
LAM = 0.5                                 # PINO residuallossweight
FAST = os.environ.get('FAST') == '1'
NTR = 16 if FAST else 128
STEPS = 30 if FAST else 8000
NTEST = 8 if FAST else 48
RES = [(16, 400)] if FAST else [(16, 400), (64, 800), (128, 1600)]
METHODS = ['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO', 'DeepONet', 'UNet']
OUT = os.path.join(ROOT, 'b1_burgers_result.txt')
L = []
def log(s): print(s, flush=True); L.append(str(s))
def save(): open(OUT, 'w', encoding='utf-8').write('\n'.join(L))


def build_trainset_nl():
    Phi, e = kl_basis(16, KMAX); PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')] = Phi, e
    rng = np.random.RandomState(0)
    U0 = np.stack([generate_multiscale_initial(16, 8, 1000 + i) for i in range(NTR)]).astype(np.float32)
    XI = rng.randn(81, NTR, KMAX).astype(np.float32).reshape(9, 9, NTR, KMAX)
    AA = np.repeat(A_GRID, 9 * NTR); SS = np.tile(np.repeat(S_GRID, NTR), 9)
    NU = make_nu(16, XI.reshape(81 * NTR, KMAX), Phi, e)
    UB = np.concatenate([U0] * 81, 0)
    Q = spectral_eigvals_torch(16, DEV, DT); sols = []; hists = []; t0 = time.time()
    for i0 in range(0, len(AA), 1024):
        fo, hb = gpu_solve_nl_batch(UB[i0:i0 + 1024], NU[i0:i0 + 1024], AA[i0:i0 + 1024],
                                    SS[i0:i0 + 1024], TN, NL_NT, 16, beta=BETA, Q=Q,
                                    device=DEV, dtype=DT, return_hist=True)
        sols.append(fo.cpu().numpy()); hists.append(hb.cpu().numpy())
    SOL = np.concatenate(sols, 0).reshape(9, 9, NTR, 17, 17)
    HIST = np.concatenate(hists, 0).reshape(9, 9, NTR, 17, 17)
    log('nonlineartraining ground truth %d  samples %.0fs (beta=%g,T=%g)' % (9 * 9 * NTR, time.time() - t0, BETA, TN))
    return U0, SOL, XI, HIST


def true_field_nl(Nx, Nt, seeds, a, s, xi, Phi, e):
    U0 = np.stack([generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
    NU = make_nu(Nx, xi, Phi, e)
    Q = spectral_eigvals_torch(Nx, DEV, DT)
    F = gpu_solve_nl_batch(U0, NU, a, s, TN, Nt, Nx, beta=BETA, Q=Q, device=DEV, dtype=DT).cpu().numpy()
    return F.reshape(len(a), -1), U0, NU


def train_pino(net, SOL, U0, XI, HIST, Qmr, tag):
    import torch.optim as opt
    op = opt.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = opt.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5); net.train()
    rng = np.random.RandomState(123); Phi, e = PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')]; t0 = time.time()

    def up(z):
        return F.interpolate(z.unsqueeze(1), size=MR + 1, mode='bicubic',
                             align_corners=False).squeeze(1)

    for st in range(STEPS + 1):
        ia, js = rng.randint(9), rng.randint(9); itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
        a, s = A_GRID[ia], S_GRID[js]
        sol = torch.tensor(SOL[ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0[itr], device=DEV, dtype=DT)
        nub = torch.tensor(make_nu(16, XI[ia, js, itr], Phi, e), device=DEV, dtype=DT)
        xb = torch.stack([u0b, nub], -1)
        ld = F.mse_loss(net(xb, (a, s)), (sol - SC.MEAN) / SC.SV)
        with torch.no_grad():
            u0f = up(u0b); nuf = up(nub); hf = up(torch.tensor(HIST[ia, js, itr], device=DEV, dtype=DT))
        xf = torch.stack([u0f, nuf], -1)
        pf = SC.MEAN + SC.SV * net(xf, (a, s))                       # 33^2 physics prediction
        pin = pf[:, 1:-1, 1:-1]
        dxu = ddx_upwind(pin, MR)
        fl = dst2_torch((Qmr ** s) * dst2_torch(pin))
        dta = (TN / NL_NT) ** a
        r = pin - hf[:, 1:-1, 1:-1] + dta * (BETA * pin * dxu + nuf[:, 1:-1, 1:-1] * fl)
        lr = F.mse_loss(r, torch.zeros_like(r)) / SC.SV ** 2
        loss = ld + LAM * lr
        op.zero_grad(); loss.backward(); op.step(); sch.step()
    net.eval(); log('  [%-7s] train %ds  params %.1fK' % (tag, time.time() - t0, n_params(net)/1e3)); return time.time()-t0


def main():
    FC.T = TN; SC.T = TN                       # linear propagator also uses T=0.015
    for Nx, _ in RES:
        Phi, e = kl_basis(Nx, KMAX); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Phi, e
    U0tr, SOL, XItrain, HIST = build_trainset_nl()
    SC.MEAN = float(SOL.mean()); SC.SV = float(SOL.std()); SC.NTR = NTR; SC.STEPS = STEPS
    PT = {Nx: build_prop_table(Nx, Nt=NL_NT) for Nx, _ in RES}
    Qmr = spectral_eigvals_torch(MR, DEV, DT)
    log('B1 nonlinearBurgers NTR=%d STEPS=%d Ntest=%d; MEAN=%.4f SV=%.4f' % (NTR, STEPS, NTEST, SC.MEAN, SC.SV))
    nets, tt = {}, {}
    def sp(name, net, kind, PT16=None):
        if kind in ('A', 'F'):
            tt[name] = SC.train_spectral(net, kind, SOL, U0tr, XItrain, PT16, name)
        else:
            tt[name] = SC.train_spatial(net, SOL, U0tr, XItrain, name)
        nets[name] = net
    sp('FrFNO', DualNet(7, seed=1), 'A', PT[16])
    sp('FNO', DualNet(2, field_dim=2, seed=2), 'F', PT[16])
    nets['PINO'] = DualNet(2, field_dim=2, seed=7); tt['PINO'] = train_pino(nets['PINO'], SOL, U0tr, XItrain, HIST, Qmr, 'PINO')
    sp('PDNO', PDNO2d(seed=6), 'F', PT[16])
    cno_net = CNOWrap(17, ch=32, use_bn=False, seed=3).to(DEV).float()
    tt['CNO'] = SC.train_spatial(cno_net, SOL, U0tr, XItrain, 'CNO', optim_type='adamw', loss_type='l1')
    nets['CNO'] = cno_net
    sp('DeepONet', DeepONet2d(w=256, seed=4).to(DEV).float(), 'sp')
    sp('UNet', UNet2d(base=48, seed=5).to(DEV).float(), 'sp')
    torch.save({k: nets[k].state_dict() for k in nets}, os.path.join(ROOT, 'b1_weights.pt'))
    rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=NTEST)
    a_te = rng.uniform(A_GRID.min(), A_GRID.max(), NTEST).astype(np.float32)
    s_te = rng.uniform(S_GRID.min(), S_GRID.max(), NTEST).astype(np.float32)
    xi_te = rng.randn(NTEST, KMAX).astype(np.float32)
    spectral_kind = {'FrFNO': 'A', 'FNO': 'F', 'PINO': 'F', 'PDNO': 'F'}
    rows = {}
    for Nx, Nt in RES:
        Phi, e = PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')]
        Ftr, U0, NU = true_field_nl(Nx, Nt, seeds, a_te, s_te, xi_te, Phi, e)
        log('\n===== resolution %d^2 (%s) =====' % (Nx + 1, 'same resolution' if Nx == 16 else 'zero-shotsuper-resolution'))
        for name in METHODS:
            if name in spectral_kind:
                em, emd, tc = SC.eval_spectral(nets[name], spectral_kind[name], Nx, U0, NU, a_te, s_te, Ftr, PT[Nx])
            else:
                em, emd, tc = SC.eval_spatial(nets[name], Nx, U0, NU, a_te, s_te, Ftr)
            rows[(Nx, name)] = em; log('  %-8s relative L2 mean%7.3f%%' % (name, em)); save()
    log('\n========= B1 summary: relative L2 mean error (%) =========')
    log('resolution    ' + ''.join('%10s' % m for m in METHODS))
    for Nx, _ in RES:
        tag = '%d^2%s' % (Nx + 1, '(train)' if Nx == 16 else '(zero-shot)')
        log('%-10s' % tag + ''.join('%9.3f%%' % rows[(Nx, m)] for m in METHODS))
    log('training time  ' + ''.join('%9.1fs' % tt[m] for m in METHODS))
    log('parameter count/K  ' + ''.join('%9.1f' % (n_params(nets[m]) / 1e3) for m in METHODS))
    np.savez(os.path.join(ROOT, 'b1_metrics.npz'),
             rows=np.array([[rows[(Nx, m)] for m in METHODS] for Nx, _ in RES]),
             methods=np.array(METHODS), res=np.array([Nx + 1 for Nx, _ in RES]),
             train_t=np.array([tt[m] for m in METHODS]))
    save()

if __name__ == '__main__':
    main()
