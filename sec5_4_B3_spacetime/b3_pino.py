# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""Enhanced space-time PINO comparison (B1 nonlinear fractional Burgers, T=0.015).
The single-step terminal operator is upgraded to a space-time solution operator (u0,nu,a,s,t)->u(t):
- TimeDualNet: same SpecConv backbone as DualNet, FiLM conditions on (a,s,t), almost identical parameter count;
- FrFNO-time: at each time level t_m it uses the corresponding analytic propagator (linear principal part, exact for any t);
- FNO-time  : a purely data-driven space-time baseline (2 channels);
- PINO-time : data plus a multi-time-level Caputo (L1) residual, with L1 coefficients identical to nl_solver.
Evaluation: terminal T at three resolutions (matched accuracy / zero-shot super-resolution) plus the mean error at intermediate 17^2 times."""
import sys, os, time
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import d_scan as DS
from d_scan import kl_basis, KMAX, PHI_CACHE
import frfno_core as FC
from frfno_core import (DynFrac2d, SpecConv2d, build_prop_table, prop_at, ub_full_batch,
                           A_GRID, S_GRID, DEV, DT)
from gpu_solver import spectral_eigvals_torch, dst2_torch
from nl_solver import gpu_solve_nl_batch, ddx_upwind
TN, BETA, NL_NT = 0.015, 3.0, 400
FC.T = TN
FAST = os.environ.get('FAST') == '1'
NTR = 16 if FAST else 128; STEPS = 30 if FAST else 8000; NTE = 8 if FAST else 48
M = 5; TM = TN*np.arange(1, M+1)/M                 # 5 equidistant physical time levels
LAM = float(os.environ.get('PINO_LAM', '0.5')); MR = 32; DA, DSNB = 0.10, 0.08
RES = [(16, 400)] if FAST else [(16, 400), (64, 800), (128, 1600)]
ORDER = ['FrFNO', 'FNO', 'PINO']
CKPT = os.path.join(ROOT, 'b3_ckpt.pt')   # incremental checkpoint, supports resume after interruption
L = []
def log(x): print(x); L.append(str(x))

class TimeDualNet(nn.Module):
    """Space-time version of DualNet: cond=(a,s,t_norm), identical backbone/width/depth."""
    def __init__(self, in_dim, modes=10, width=48, n_layers=4, field_dim=2, pad=4, emb=32, seed=0):
        super().__init__(); torch.manual_seed(seed); self.pad = pad; self.nl = n_layers
        self.act = nn.LeakyReLU(); self.frac = DynFrac2d(field_dim, modes)
        lifted = in_dim + 2 + self.frac.extra + 3   # the final 3 = conditioning fields a,s,t
        self.fc0 = nn.Sequential(nn.Linear(lifted, 128), self.act, nn.Linear(128, width))
        self.sp = nn.ModuleList([SpecConv2d(width, width, modes) for _ in range(n_layers)])
        self.cv = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.enc = nn.Sequential(nn.Linear(3, emb), nn.GELU(), nn.Linear(emb, emb))
        self.fg = nn.ModuleList([nn.Linear(emb, width) for _ in range(n_layers)])
        self.fb = nn.ModuleList([nn.Linear(emb, width) for _ in range(n_layers)])
        self.q = nn.Sequential(nn.Linear(width, 128), self.act, nn.Linear(128, 1)); self.to(DEV)
    def forward(self, x, cond):
        a, s, tn = float(cond[0]), float(cond[1]), float(cond[2]); B, H, W, _ = x.shape
        gx = torch.linspace(0, 1, W, device=x.device).view(1, 1, W, 1).repeat(B, H, 1, 1)
        gy = torch.linspace(0, 1, H, device=x.device).view(1, 1, W, 1).repeat(B, H, 1, 1)
        x = torch.cat([torch.cat([gy, gx], -1), x], -1)
        h = self.frac(x, a)
        ach = torch.cat([torch.full((B, H, W, 1), a, device=x.device),
                         torch.full((B, H, W, 1), s, device=x.device),
                         torch.full((B, H, W, 1), tn, device=x.device)], -1)
        h = self.fc0(torch.cat([h, ach], -1)).permute(0, 3, 1, 2)
        e = self.enc(torch.tensor([[a, s, tn]], device=DEV, dtype=DT).repeat(B, 1))
        if self.pad > 0: h = F.pad(h, [0, self.pad, 0, self.pad])
        for i in range(self.nl):
            h = self.sp[i](h) + self.cv[i](h)
            g = (1+self.fg[i](e)).unsqueeze(-1).unsqueeze(-1); b = self.fb[i](e).unsqueeze(-1).unsqueeze(-1)
            h = g*h + b
            if i != self.nl-1: h = self.act(h)
        if self.pad > 0: h = h[..., :-self.pad, :-self.pad]
        return self.q(h.permute(0, 2, 3, 1)).squeeze(-1)

def n_params(net): return sum(p.numel() for p in net.parameters())

def build_trainset():
    Phi, e = kl_basis(16, KMAX); PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')] = Phi, e
    rng = np.random.RandomState(0)
    U0 = np.stack([DS.generate_multiscale_initial(16, 8, 1000+i) for i in range(NTR)]).astype(np.float32)
    XI = rng.randn(81, NTR, KMAX).astype(np.float32).reshape(9, 9, NTR, KMAX)
    AA = np.repeat(A_GRID, 9*NTR); SS = np.tile(np.repeat(S_GRID, NTR), 9)
    NU = DS.make_nu(16, XI.reshape(81*NTR, KMAX), Phi, e)
    UB = np.concatenate([U0]*81, 0); Q = spectral_eigvals_torch(16, DEV, DT)
    snap = [int(round((m+1)*NL_NT/M)) for m in range(M)]
    t0 = time.time()
    fo, _, SN = gpu_solve_nl_batch(UB, NU, AA, SS, TN, NL_NT, 16, beta=BETA, Q=Q,
                                   device=DEV, dtype=DT, return_hist=True, snap_idx=snap)
    SOL = SN.cpu().numpy().reshape(M, 9, 9, NTR, 17, 17)
    log('space-time training ground-truth snapshots, %d levels x %d samples, %.0fs' % (M, 9*9*NTR, time.time()-t0))
    return U0, SOL, XI

def build_pt_t(nx, nt):
    """Build one propagator table for each of the M time levels (linear, arbitrary t)."""
    PTm = []
    for m in range(M):
        FC.T = float(TM[m]); Ntm = max(8, int(round(nt*(m+1)/M)))
        PTm.append(build_prop_table(nx, Nt=Ntm))
    FC.T = TN; return PTm

def train_data(net, kind, SOL, U0, XI, PT16, tag):
    import torch.optim as opt
    op = opt.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = opt.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5); net.train(); t0 = time.time()
    Phi, e = PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')]; rng = np.random.RandomState(123)
    for st in range(STEPS+1):
        ia, js = rng.randint(9), rng.randint(9); m = rng.randint(M); itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
        a, s, tn = A_GRID[ia], S_GRID[js], float((m+1)/M)
        sol = torch.tensor(SOL[m, ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0[itr], device=DEV, dtype=DT)
        nub = torch.tensor(DS.make_nu(16, XI[ia, js, itr], Phi, e), device=DEV, dtype=DT)
        if kind == 'A':
            gs = [prop_at(a, s, *PT16[m])] + [prop_at(*z, *PT16[m]) for z in
                  [(a-DA, s), (a+DA, s), (a, s-DSNB), (a, s+DSNB)]]
            ubs = [ub_full_batch(U0[itr], g, 16) for g in gs]
            xb = torch.stack([u0b, ubs[0], nub, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            pred = ubs[0] + SV*net(xb, (a, s, tn))
            loss = F.mse_loss(pred, sol)
        else:
            xb = torch.stack([u0b, nub], -1)
            loss = F.mse_loss(net(xb, (a, s, tn)), (sol-MEAN)/SV)
        op.zero_grad(); loss.backward(); op.step(); sch.step()
    net.eval(); log('  [%-6s] train %ds params %.1fK' % (tag, time.time()-t0, n_params(net)/1e3)); return time.time()-t0

def l1_coeffs(alpha, m):
    jj = torch.arange(m+1, device=DEV, dtype=DT); b = (jj+1.)**(1-alpha)-jj**(1-alpha)
    return b

def train_pino_time(net, SOL, U0, XI, tag):
    import torch.optim as opt
    op = opt.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = opt.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5); net.train(); t0 = time.time()
    Phi, e = PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')]; rng = np.random.RandomState(123)
    Qmr = spectral_eigvals_torch(MR, DEV, DT); tau = TN/M
    def up(z): return F.interpolate(z.unsqueeze(1), size=MR+1, mode='bicubic', align_corners=False).squeeze(1)
    for st in range(STEPS+1):
        ia, js = rng.randint(9), rng.randint(9); m = rng.randint(1, M); itr = torch.randperm(NTR, device=DEV)[:32].cpu().numpy()
        a, s = A_GRID[ia], S_GRID[js]
        sol = torch.tensor(SOL[m, ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0[itr], device=DEV, dtype=DT)
        nub = torch.tensor(DS.make_nu(16, XI[ia, js, itr], Phi, e), device=DEV, dtype=DT)
        xb = torch.stack([u0b, nub], -1)
        ld = F.mse_loss(net(xb, (a, s, (m+1)/M)), (sol-MEAN)/SV)
        loss = ld
        if LAM > 0:
            with torch.no_grad():
                u0f, nuf = up(u0b), up(nub)
            pf = []
            for j in range(m+1):
                pj = MEAN+SV*net(torch.stack([u0f, nuf], -1), (a, s, (j+1)/M))
                pf.append(pj[:, 1:-1, 1:-1])
            b = l1_coeffs(a, m); dta = tau**a
            base = b[m-1]*u0f[:, 1:-1, 1:-1]
            for j in range(1, m):
                base = base + (b[m-j-1]-b[m-j])*pf[j]
            cap = (b[0]*pf[m]-base)/dta
            dxu = ddx_upwind(pf[m], MR); fl = dst2_torch((Qmr**s)*dst2_torch(pf[m]))
            r = cap + BETA*pf[m]*dxu + nuf[:, 1:-1, 1:-1]*fl
            lr = F.mse_loss(r, torch.zeros_like(r))/SV**2
            loss = ld + LAM*lr
        loss.backward(); op.step(); sch.step(); op.zero_grad()
    net.eval(); log('  [%-6s] train %ds params %.1fK (space-timeCaputocollocation)' % (tag, time.time()-t0, n_params(net)/1e3)); return time.time()-t0

@torch.no_grad()
def eval_final(net, kind, Nx, U0, NU, a, s, Ftrue, PTm):
    errs = []
    for i in range(len(a)):
        u0t = torch.tensor(U0[i:i+1], device=DEV, dtype=DT); nut = torch.tensor(NU[i:i+1], device=DEV, dtype=DT)
        aa, ss = float(a[i]), float(s[i]); tn = 1.0
        if kind == 'A':
            gs = [prop_at(aa, ss, *PTm[-1])] + [prop_at(*z, *PTm[-1]) for z in
                  [(aa-DA, ss), (aa+DA, ss), (aa, ss-DSNB), (aa, ss+DSNB)]]
            ubs = [ub_full_batch(U0[i:i+1], g, Nx) for g in gs]
            xb = torch.stack([u0t, ubs[0], nut, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            p = (ubs[0]+SV*net(xb, (aa, ss, tn))).cpu().numpy().ravel()
        else:
            xb = torch.stack([u0t, nut], -1)
            p = (MEAN+SV*net(xb, (aa, ss, tn))).cpu().numpy().ravel()
        errs.append(np.linalg.norm(p-Ftrue[i])/(np.linalg.norm(Ftrue[i])+1e-30))
    return 100*np.mean(errs)

def main():
    global MEAN, SV
    for Nx, _ in RES:
        Phi, e = kl_basis(Nx, KMAX); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Phi, e
    U0, SOL, XI = build_trainset()
    MEAN = float(SOL[-1].mean()); SV = float(SOL[-1].std())
    log('MEAN=%.4f SV=%.4f (terminal statistics)' % (MEAN, SV))
    PT16 = build_pt_t(16, NL_NT)
    ck = torch.load(CKPT, map_location=DEV) if os.path.exists(CKPT) else {}
    nets, tt = {}, {}
    def build_or_load(name, ctor, trainfn):
        net = ctor().to(DEV); nets[name] = net
        if name in ck:
            net.load_state_dict(ck[name]); tt[name] = 0.0; log('  restore %s from checkpoint, skip training (restored)' % name)
        else:
            tt[name] = trainfn(net); ck[name] = net.state_dict()
            torch.save(ck, CKPT)                       # write to disk immediately after each model finishes training
    build_or_load('FrFNO', lambda: TimeDualNet(7, field_dim=2, seed=1),
                  lambda n: train_data(n, 'A', SOL, U0, XI, PT16, 'FrFNO'))
    build_or_load('FNO', lambda: TimeDualNet(2, field_dim=0, seed=2),
                  lambda n: train_data(n, 'F', SOL, U0, XI, PT16, 'FNO'))
    # PINO checkpoint keys carry the LAM tag; each residual weight is stored separately for a fair ablation
    pkey = 'PINO_lam%g' % LAM; pnet = TimeDualNet(2, field_dim=0, seed=7); nets['PINO'] = pnet
    if pkey in ck:
        pnet.load_state_dict(ck[pkey]); log('  restore PINO from checkpoint (LAM=%g)' % LAM)
    else:
        tt['PINO'] = train_pino_time(pnet, SOL, U0, XI, 'PINO'); ck[pkey] = pnet.state_dict(); torch.save(ck, CKPT)
    torch.save({k: nets[k].state_dict() for k in nets}, os.path.join(ROOT, 'b3_weights_lam%g.pt') % LAM)
    rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=NTE)
    at = rng.uniform(A_GRID.min(), A_GRID.max(), NTE).astype(np.float32)
    st = rng.uniform(S_GRID.min(), S_GRID.max(), NTE).astype(np.float32)
    xit = rng.randn(NTE, KMAX).astype(np.float32); rows = {}; kinds = {'FrFNO': 'A', 'FNO': 'F', 'PINO': 'F'}
    for Nx, Nt in RES:
        Phi, e = PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')]
        snap = [int(round(Nt))]; Q = spectral_eigvals_torch(Nx, DEV, DT)
        U0e = np.stack([DS.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
        NUe = DS.make_nu(Nx, xit, Phi, e)
        Fm = gpu_solve_nl_batch(U0e, NUe, at, st, TN, Nt, Nx, beta=BETA, Q=Q, device=DEV, dtype=DT,
                                snap_idx=snap)[-1][-1].cpu().numpy().reshape(NTE, -1)
        PTm = build_pt_t(Nx, Nt)
        tag = '%d^2(%s)' % (Nx+1, 'train' if Nx == 16 else 'zero-shot')
        log('\n== terminal %s ==' % tag)
        for m in ORDER:
            rows[(Nx, m)] = eval_final(nets[m], kinds[m], Nx, U0e, NUe, at, st, Fm, PTm)
            log('  %-6s %.3f%%' % (m, rows[(Nx, m)]))
    log('\n===== space-time terminal relL2(%) =====')
    log('resolution  ' + ''.join('%10s' % m for m in ORDER))
    for Nx, _ in RES:
        log('%-8s' % ('%d^2' % (Nx+1)) + ''.join('%9.3f%%' % rows[(Nx, m)] for m in ORDER))
    open(os.path.join(ROOT, 'b3_result_lam%g.txt') % LAM, 'w', encoding='utf-8').write('\n'.join(L))

if __name__ == '__main__':
    main()
