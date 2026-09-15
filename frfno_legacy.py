# -*- coding: utf-8 -*-
"""
Legacy forward uncertainty-quantification comparison for a fractional PDE with random orders (alpha, s) and a 2-D random input
PDE: ^C D_t^a u = -nu(x,y)(-Delta)^s u, T=0.01, variable log-normal nu
Five method families share one reference and one fully-implicit GPU solver:
  sampling : MC reference (large Sobol sample), QMC (first N Sobol points, nested)
  spectral : gPC (truncated-normal Gauss tensor collocation / PCE)
  sparse   : SC (2-D Smolyak built from 1-D Gauss rules)
  surrogate: GP/Kriging (RBF GP on PCA coefficients, Sobol training points)
  neural operators (trained at 17^2, zero-shot inference at target resolution):
        FrFNO (strong analytic-propagator injection), naive FNO, deep ensemble (K=5)
Resolutions 65^2/129^2/257^2; metrics: mean/std relative L2, std spatial correlation, 95% coverage, cost. Retained only for the theory-closure Exp.3-5; main results use frfno_core.py
"""
import sys, os, time
_RDIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import truncnorm, norm
from scipy.stats.qmc import Sobol
sys.path.insert(0, os.path.join(_RDIR, 'fractional_pde_study'))
sys.path.insert(0, _RDIR)
from solver import generate_multiscale_initial
from solver_variable import smooth_nu_field
from solver_spacetime_fractional import spectral_eigvals, _l1_b
from gpu_solver import gpu_solve_imex_batch, dst2_torch, spectral_eigvals_torch

DEV = torch.device('cuda'); DT = torch.float32; T = 0.01
OUT = os.path.join(_RDIR, 'legacy_compare_result.txt')
L = []
def log(s): print(s, flush=True); L.append(str(s))
def save(): open(OUT, 'w', encoding='utf-8').write('\n'.join(L))

A_GRID = np.linspace(0.55, 0.95, 9); S_GRID = np.linspace(0.35, 0.78, 9)
DA, DS = 0.10, 0.08; NX_TR = 16
FAST = os.environ.get('FAST') == '1'
NTR = 16 if FAST else 128; STEPS = 30 if FAST else 8000
K_ENS = 2 if FAST else 5
RNG_A = dict(mu=.75, sd=.10, lo=.60, hi=.95)
RNG_S = dict(mu=.60, sd=.12, lo=.35, hi=.78)
RES = [(64, 800, 128 if FAST else 256), (128, 1600, 128),
       (256, 1600, 128 if FAST else 128)]   # (Nx, Nt, Nref); Nref is a power of two for balanced Sobol
NTE = 2 if FAST else 3; M_INF = 32 if FAST else 128

# ============ networks (identical to the main FrFNO backbone) ============
class DynFrac2d(nn.Module):
    def __init__(self, field_dim, modes, kind='I'):
        super().__init__(); self.fd = field_dim; self.modes = modes
        self.sign = -1.0 if kind == 'I' else 1.0; self.cache = {}
    def _basis(self, H, W, dev, dt):
        if (H, W) not in self.cache:
            kx = torch.fft.fftfreq(H, d=1./H, device=dev, dtype=dt)
            ky = torch.fft.rfftfreq(W, d=1./W, device=dev, dtype=dt)
            KX, KY = torch.meshgrid(kx, ky, indexing='ij'); K = torch.sqrt(KX**2+KY**2)
            lk = torch.where(K > 0, torch.log(K.clamp(min=1.)), torch.zeros_like(K))
            mx = torch.arange(H, device=dev).view(H, 1); my = torch.arange(ky.shape[0], device=dev).view(1, -1)
            low = (((mx < self.modes) | (mx > H-self.modes)) & (my < self.modes) & (K > 0)).to(dt)
            self.cache[(H, W)] = (lk, low)
        return self.cache[(H, W)]
    def _one(self, f, a):
        H, W = f.shape[-2], f.shape[-1]; lk, low = self._basis(H, W, f.device, f.dtype)
        Ff = torch.fft.rfft2(f, dim=(-2, -1))
        return torch.fft.irfft2(Ff*torch.exp(self.sign*a*lk)*low, s=(H, W), dim=(-2, -1))
    def forward(self, x, a):
        feats = [x]
        for c in range(self.fd): feats.append(self._one(x[..., c], a).unsqueeze(-1))
        return torch.cat(feats, -1)
    @property
    def extra(self): return self.fd

class SpecConv2d(nn.Module):
    def __init__(self, ci, co, m):
        super().__init__(); self.m = m
        self.w1 = nn.Parameter((1./(ci*co))*torch.rand(ci, co, m, m, dtype=torch.cfloat))
        self.w2 = nn.Parameter((1./(ci*co))*torch.rand(ci, co, m, m, dtype=torch.cfloat))
    def forward(self, x):
        B, _, n1, n2 = x.shape; m = self.m; xf = torch.fft.rfft2(x)
        o = torch.zeros(B, self.w1.shape[1], n1, n2//2+1, device=x.device, dtype=torch.cfloat)
        o[:, :, :m, :m] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :m, :m], self.w1)
        o[:, :, -m:, :m] = torch.einsum('bixy,ioxy->boxy', xf[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(o, s=(n1, n2))

class DualNet(nn.Module):
    def __init__(self, in_dim, modes=10, width=48, n_layers=4, field_dim=1, pad=4, emb=32, seed=0):
        super().__init__()
        torch.manual_seed(seed); self.pad = pad; self.nl = n_layers; self.act = nn.LeakyReLU()
        self.frac = DynFrac2d(field_dim, modes)
        lifted = in_dim + 2 + self.frac.extra + 2
        self.fc0 = nn.Sequential(nn.Linear(lifted, 128), self.act, nn.Linear(128, width))
        self.sp = nn.ModuleList([SpecConv2d(width, width, modes) for _ in range(n_layers)])
        self.cv = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.enc = nn.Sequential(nn.Linear(2, emb), nn.GELU(), nn.Linear(emb, emb))
        self.fg = nn.ModuleList([nn.Linear(emb, width) for _ in range(n_layers)])
        self.fb = nn.ModuleList([nn.Linear(emb, width) for _ in range(n_layers)])
        self.q = nn.Sequential(nn.Linear(width, 128), self.act, nn.Linear(128, 1)); self.to(DEV)
    def forward(self, x, cond):
        a, s = float(cond[0]), float(cond[1]); B, H, W, _ = x.shape
        gx = torch.linspace(0, 1, W, device=x.device).view(1, 1, W, 1).repeat(B, H, 1, 1)
        gy = torch.linspace(0, 1, H, device=x.device).view(1, H, 1, 1).repeat(B, 1, W, 1)
        x = torch.cat([torch.cat([gy, gx], -1), x], -1)
        h = self.frac(x, a)
        ach = torch.cat([torch.full((B, H, W, 1), a, device=x.device),
                         torch.full((B, H, W, 1), s, device=x.device)], -1)
        h = self.fc0(torch.cat([h, ach], -1)).permute(0, 3, 1, 2)
        e = self.enc(torch.tensor([[a, s]], device=DEV, dtype=DT).repeat(B, 1))
        if self.pad > 0: h = F.pad(h, [0, self.pad, 0, self.pad])
        for i in range(self.nl):
            h = self.sp[i](h) + self.cv[i](h)
            g = (1+self.fg[i](e)).unsqueeze(-1).unsqueeze(-1); b = self.fb[i](e).unsqueeze(-1).unsqueeze(-1)
            h = g*h + b
            if i != self.nl-1: h = self.act(h)
        if self.pad > 0: h = h[..., :-self.pad, :-self.pad]
        return self.q(h.permute(0, 2, 3, 1)).squeeze(-1)

# ============ analytic propagator table / FrFNO features ============
def build_prop_table(Nx, na=41, ns=41, Nt=400):
    Q = spectral_eigvals(Nx); n = Nx-1
    AD = np.linspace(0.45, 1.05, na); SD = np.linspace(0.25, 0.90, ns)
    qf = torch.tensor(Q.reshape(-1), device=DEV, dtype=DT); PT = np.zeros((na, ns, n, n), np.float32)
    for ia, al in enumerate(AD):
        b = torch.tensor(_l1_b(al, Nt), device=DEV, dtype=DT)
        lam = qf[None] ** torch.tensor(SD, device=DEV, dtype=DT)[:, None]
        dta = (T/Nt)**al; b0 = b[0]; hist = torch.empty((Nt, ns, n*n), device=DEV, dtype=DT); hist[0] = 1.0; v = hist[0]
        for k in range(1, Nt+1):
            rhs = b[k-1]*hist[0]
            if k > 1:
                jj = torch.arange(1, k, device=DEV); rhs = rhs + torch.einsum('j,jxy->xy', b[k-jj-1]-b[k-jj], hist[1:k])
            v = rhs/(b0+dta*lam)
            if k < Nt: hist[k] = v
        PT[ia] = v.reshape(ns, n, n).cpu().numpy(); del hist
    torch.cuda.empty_cache(); return AD, SD, PT

def prop_at(alpha, s, AD, SD, PT):
    ia = np.clip(np.searchsorted(AD, alpha)-1, 0, len(AD)-2)
    js = np.clip(np.searchsorted(SD, s)-1, 0, len(SD)-2)
    ta = (alpha-AD[ia])/(AD[ia+1]-AD[ia]); ts = (s-SD[js])/(SD[js+1]-SD[js])
    return ((1-ta)*(1-ts)*PT[ia,js]+ta*(1-ts)*PT[ia+1,js]
            +(1-ta)*ts*PT[ia,js+1]+ta*ts*PT[ia+1,js+1])

def ub_full_batch(u0_full, g, Nx):
    ui = torch.as_tensor(u0_full[:, 1:Nx, 1:Nx], device=DEV, dtype=DT)
    gt = torch.as_tensor(g, device=DEV, dtype=DT)
    inner = dst2_torch(gt[None]*dst2_torch(ui))
    out = torch.zeros(ui.shape[0], Nx+1, Nx+1, device=DEV, dtype=DT); out[:, 1:Nx, 1:Nx] = inner; return out

# ============ random-input utilities ============
def tn_ppf(u, p):  # [0,1] -> truncated-normal quantile
    a, b = (p['lo']-p['mu'])/p['sd'], (p['hi']-p['mu'])/p['sd']
    return truncnorm.ppf(u, a, b, loc=p['mu'], scale=p['sd'])
def tn_pdf(x, p):
    a, b = (p['lo']-p['mu'])/p['sd'], (p['hi']-p['mu'])/p['sd']
    return truncnorm.pdf(x, a, b, loc=p['mu'], scale=p['sd'])

def sobol_points(n, seed=42):
    sb = Sobol(d=2, scramble=True, seed=seed).random(n)
    return tn_ppf(sb[:, 0], RNG_A), tn_ppf(sb[:, 1], RNG_S)

def mc_points(n, seed=7):
    rng = np.random.RandomState(seed); u = rng.rand(n, 2)
    return tn_ppf(u[:, 0], RNG_A), tn_ppf(u[:, 1], RNG_S)

def discrete_gauss_1d(p, n, ng=4001):
    """n-point Gauss rule under a truncated-normal measure (Golub-Welsch, discrete Stieltjes)."""
    z = np.linspace(p['lo'], p['hi'], ng); w = tn_pdf(z, p); w = w/w.sum()
    a = np.zeros(n); b = np.zeros(n)
    # three-term recurrence coefficients (discrete inner product)
    mom = lambda f: np.sum(w*f)
    P_prev = np.ones_like(z); a[0] = mom(z*P_prev**2)/mom(P_prev**2)
    P = z - a[0]; b[0] = 0.
    for k in range(1, n):
        nm = mom(P**2)
        a[k] = mom(z*P**2)/nm
        b[k] = nm/mom(P_prev**2)
        Pn = (z-a[k])*P - b[k]*P_prev; P_prev, P = P, Pn
    J = np.diag(a) + np.diag(np.sqrt(b[1:]), 1) + np.diag(np.sqrt(b[1:]), -1)
    x, V = np.linalg.eigh(J); wi = V[0, :]**2
    return x, wi

def tensor_gauss(n_per):
    xa, wa = discrete_gauss_1d(RNG_A, n_per); xs, ws = discrete_gauss_1d(RNG_S, n_per)
    AA, SS = np.meshgrid(xa, xs, indexing='ij'); WA, WS = np.meshgrid(wa, ws, indexing='ij')
    return AA.ravel(), SS.ravel(), (WA*WS).ravel()

def smolyak_2d(L):
    """2-D Smolyak sparse quadrature; level i uses the (i+1)-point 1-D Gauss rule; nearby nodes are merged."""
    nodes = {}
    for i1 in range(1, L+1):
        for i2 in range(1, L+1):
            lev = i1+i2
            if lev < L+1 or lev > 2*L: continue
            coef = (-1)**(2*L-lev) * _comb(1, 2*L-lev)
            x1, w1 = discrete_gauss_1d(RNG_A, i1+1)
            x2, w2 = discrete_gauss_1d(RNG_S, i2+1)
            for a_, wa_ in zip(x1, w1):
                for s_, ws_ in zip(x2, w2):
                    key = (round(a_, 9), round(s_, 9))
                    nodes[key] = nodes.get(key, 0.) + coef*wa_*ws_
    aa = np.array([k[0] for k in nodes]); ss = np.array([k[1] for k in nodes]); w = np.array(list(nodes.values()))
    return aa, ss, w
def _comb(n, k):
    from math import comb
    return comb(n, k) if 0 <= k <= n else 0

# ============ small-sample GP (RBF GP on PCA coefficients) ============
class PcaGP:
    def __init__(self, n_comp=20, ls=0.35, sn=1e-4):
        self.r = n_comp; self.ls = ls; self.sn = sn
    def _K(self, X, Z=None):
        Z = X if Z is None else Z
        d = ((X[:, None, :]-Z[None, :, :])/self.ls)**2
        return np.exp(-0.5*d.sum(-1))
    def fit(self, X, Y):
        self.mu = Y.mean(0); Yc = Y-self.mu
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
        self.r = min(self.r, len(S)); self.V = Vt[:self.r]
        self.C = Yc @ self.V.T
        self.X = X.copy(); K = self._K(X) + self.sn*np.eye(len(X))
        self.Lc = np.linalg.cholesky(K + 1e-8*np.eye(len(X)))
        self.alpha = np.linalg.solve(self.Lc.T, np.linalg.solve(self.Lc, self.C))
    def predict(self, Xs):
        Ks = self._K(self.X, Xs)
        Cs = Ks.T @ self.alpha
        return self.mu[None, :] + Cs @ self.V

# ============ moments / metrics ============
def wmom(F, w):
    w = w/np.sum(w); m = (w[:, None]*F).sum(0); e2 = (w[:, None]*F**2).sum(0)
    return m, np.maximum(e2-m*m, 0)
def mom(F):
    return F.mean(0), np.maximum(F.var(0), 0)
def err_of(m, v, mr, vr):
    s, sr = np.sqrt(v), np.sqrt(vr)
    me = np.linalg.norm(m-mr)/(np.linalg.norm(mr)+1e-12)
    se = np.linalg.norm(s-sr)/(np.linalg.norm(sr)+1e-12)
    cc = np.corrcoef(s, sr)[0, 1] if np.std(s) > 0 and np.std(sr) > 0 else 0.
    return me, se, cc
def coverage(m, v, holdout):
    lo = m-1.96*np.sqrt(v); hi = m+1.96*np.sqrt(v)
    return float(np.mean((holdout >= lo) & (holdout <= hi)))

# ============ 17^2 training data and training ============
def build_trainset():
    log(f'\n=== build 17^2 training data, {NTR} fields x 9x9 order grid (batched fully-implicit GPU) ===')
    U0 = np.stack([generate_multiscale_initial(NX_TR, 8, 1000+i) for i in range(NTR)]).astype(np.float32)
    NU = np.stack([smooth_nu_field(NX_TR, 2000+i, .4, 4) for i in range(NTR)]).astype(np.float32)
    NU = NU/NU.mean(axis=(1, 2), keepdims=True)
    Q = spectral_eigvals_torch(NX_TR, DEV, DT)
    NA, NS = len(A_GRID), len(S_GRID)
    AA = np.repeat(A_GRID, NS*NTR); SS = np.tile(np.repeat(S_GRID, NTR), NA)
    UB = np.concatenate([U0]*NA*NS, 0); NB = np.concatenate([NU]*NA*NS, 0)
    outs = []; t0 = time.time()
    for i0 in range(0, len(AA), 1024):
        outs.append(gpu_solve_imex_batch(UB[i0:i0+1024], NB[i0:i0+1024], AA[i0:i0+1024], SS[i0:i0+1024],
                                         T, 400, NX_TR, Q=Q, device=DEV, dtype=DT).cpu().numpy())
    SOL = np.concatenate(outs, 0).reshape(NA, NS, NTR, NX_TR+1, NX_TR+1)
    log(f'  training ground truth {NA*NS*NTR}  samples {time.time()-t0:.0f}s')
    return U0, NU, SOL

def train(net, kind, SOL, U0tr, NUtr, AD, SD, PT, tag):
    opt = torch.optim.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, 5e-5)
    net.train(); t0 = time.time()
    for st in range(STEPS+1):
        itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
        ia = np.random.randint(len(A_GRID)); js = np.random.randint(len(S_GRID))
        a, s = A_GRID[ia], S_GRID[js]; sol = torch.tensor(SOL[ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0tr[itr], device=DEV, dtype=DT)
        nub = torch.tensor(NUtr[itr], device=DEV, dtype=DT)
        if kind == 'A':
            gs = [prop_at(a, s, AD, SD, PT)] + [prop_at(*z, AD, SD, PT) for z in
                  [(a-DA, s), (a+DA, s), (a, s-DS), (a, s+DS)]]
            ubs = [ub_full_batch(U0tr[itr], g, NX_TR) for g in gs]
            xb = torch.stack([u0b, ubs[0], nub, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            tg = (sol-ubs[0])/SV
        else:
            xb = torch.stack([u0b, nub], -1); tg = (sol-MEAN)/SV
        loss = F.mse_loss(net(xb, (a, s)), tg)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval(); log(f'  [{tag}] trained in {time.time()-t0:.0f}s'); return time.time()-t0

@torch.no_grad()
def infer(net, kind, u0s, nus, aa, ss, Nx, AD, SD, PT):
    """u0s/nus: (nte,H,W); forward each of M (a,s) -> list[nte] of (M,H,W) physical fields."""
    nte = len(u0s); per = [[] for _ in range(nte)]
    u0t = torch.tensor(u0s, device=DEV, dtype=DT); nut = torch.tensor(nus, device=DEV, dtype=DT)
    for m in range(len(aa)):
        a, s = float(aa[m]), float(ss[m])
        if kind == 'A':
            gs = [prop_at(a, s, AD, SD, PT)] + [prop_at(*z, AD, SD, PT) for z in
                  [(a-DA, s), (a+DA, s), (a, s-DS), (a, s+DS)]]
            ubs = [ub_full_batch(u0s, g, Nx) for g in gs]
            xb = torch.stack([u0t, ubs[0], nut, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            o = net(xb, (a, s)).cpu().numpy(); phys = ubs[0].cpu().numpy()+SV*o
        else:
            xb = torch.stack([u0t, nut], -1)
            o = net(xb, (a, s)).cpu().numpy(); phys = MEAN+SV*o
        for i in range(nte): per[i].append(phys[i])
    return [np.stack(p) for p in per]

def solve_points(Nx, Nt, u0, nu, aa, ss, Q, bs=16):
    """Batched solve of one field over a set of (alpha,s) -> (len,D) and per-sample wall time."""
    aa = np.asarray(aa); ss = np.asarray(ss); B = len(aa); outs = []; t0 = time.time()
    for i0 in range(0, B, bs):
        o = gpu_solve_imex_batch(np.repeat(u0[None], min(bs, B-i0), 0),
                                 np.repeat(nu[None], min(bs, B-i0), 0),
                                 aa[i0:i0+bs], ss[i0:i0+bs], T, Nt, Nx, Q=Q,
                                 device=DEV, dtype=DT).cpu().numpy()
        outs.append(o.reshape(o.shape[0], -1))
    wall = time.time()-t0
    return np.concatenate(outs, 0), wall/max(B, 1)

# ============ main driver ============
def main():
    global SV, MEAN
    t_train0 = time.time()
    U0tr, NUtr, SOL = build_trainset()
    SV = SOL.std(); MEAN = SOL.mean()
    ADtr, SDtr, PTtr = build_prop_table(NX_TR, Nt=400)
    netA = DualNet(7, seed=1); tA = train(netA, 'A', SOL, U0tr, NUtr, ADtr, SDtr, PTtr, 'FrFNO')
    netsB = []; tBtot = 0
    for k in range(K_ENS):
        nb = DualNet(2, seed=10+k); tBtot += train(nb, 'B', SOL, U0tr, NUtr, ADtr, SDtr, PTtr, f'Naive#{k}'); netsB.append(nb)
    t_train = time.time()-t_train0
    log(f'  total neural-operator training {t_train:.0f}s (FrFNO {tA:.0f}s + ensemble x{K_ENS} {tBtot:.0f}s)')

    rows = []
    for (Nx, Nt, Nref) in RES:
        log(f'\n======== Nx={Nx} ({Nx+1}^2) Nt={Nt}, MC reference N={Nref} ========')
        tpb = time.time(); AD, SD, PT = build_prop_table(Nx, Nt=400); log(f'  propagator table (Nt=400, matches training) {time.time()-tpb:.0f}s')
        Q = spectral_eigvals_torch(Nx, DEV, DT)
        ref_a, ref_s = sobol_points(Nref)
        gpc_a, gpc_s, gpc_w = tensor_gauss(5)
        sc_a, sc_s, sc_w = smolyak_2d(3)
        inf_a, inf_s = mc_points(M_INF, seed=20240908)
        agg = {}
        for i in range(NTE):
            u0 = generate_multiscale_initial(Nx, 8, 500000+i).astype(np.float32)
            nu = smooth_nu_field(Nx, 600000+i, .4, 4).astype(np.float32); nu = nu/nu.mean()
            # solver pool: reference (QMC/GP nested in it) + gPC + SC unique nodes
            ua = np.concatenate([ref_a, gpc_a, sc_a]); us = np.concatenate([ref_s, gpc_s, sc_s])
            Fpool, csolve = solve_points(Nx, Nt, u0, nu, ua, us, Q, bs=8 if Nx >= 256 else 16)
            Fref = Fpool[:Nref]
            Fgpc = Fpool[Nref:Nref+len(gpc_a)]
            Fsc = Fpool[Nref+len(gpc_a):]
            mr, vr = mom(Fref)
            res = {}
            # sampling
            for Nq in [32, 64]:
                if Nq <= Nref:
                    m, v = mom(Fref[:Nq]); res[f'QMC{Nq}'] = (*err_of(m, v, mr, vr), coverage(m, v, Fref), Nq*csolve)
            # gPC / SC (weighted)
            m, v = wmom(Fgpc, gpc_w); res['gPC(25)'] = (*err_of(m, v, mr, vr), coverage(m, v, Fref), len(gpc_a)*csolve)
            m, v = wmom(Fsc, sc_w);  res['SC-Smol'] = (*err_of(m, v, mr, vr), coverage(m, v, Fref), len(sc_a)*csolve)
            # GP (trained on first 25 reference points, inferred on M_INF random samples)
            ng = 25
            Xtr = np.stack([(ref_a[:ng]-RNG_A['lo'])/(RNG_A['hi']-RNG_A['lo']),
                            (ref_s[:ng]-RNG_S['lo'])/(RNG_S['hi']-RNG_S['lo'])], 1)
            Xinf = np.stack([(inf_a-RNG_A['lo'])/(RNG_A['hi']-RNG_A['lo']),
                             (inf_s-RNG_S['lo'])/(RNG_S['hi']-RNG_S['lo'])], 1)
            gp = PcaGP(n_comp=20); gp.fit(Xtr, Fref[:ng]); Fgp = gp.predict(Xinf)
            m, v = mom(Fgp); res['GP/Kriging'] = (*err_of(m, v, mr, vr), coverage(m, v, Fref), ng*csolve)
            # neural operators
            t0 = time.time()
            pA = infer(netA, 'A', u0[None], nu[None], inf_a, inf_s, Nx, AD, SD, PT)[0].reshape(len(inf_a), -1)
            tAinf = (time.time()-t0)
            m, v = mom(pA); res['FrFNO'] = (*err_of(m, v, mr, vr), coverage(m, v, Fref), tAinf)
            pEns = []; t0b = time.time()
            for k, nb in enumerate(netsB):
                p = infer(nb, 'B', u0[None], nu[None], inf_a, inf_s, Nx, AD, SD, PT)[0].reshape(len(inf_a), -1); pEns.append(p)
            tBinf = time.time()-t0b
            pEns = np.stack(pEns).reshape(K_ENS, len(inf_a), -1)  # (K,M,D)
            mB, vB = mom(pEns[0]); res['Naive'] = (*err_of(mB, vB, mr, vr), coverage(mB, vB, Fref), tBinf/K_ENS)
            mk = pEns.mean(1)                 # (K,D) ensemble mean of members
            ale = pEns.var(1).mean(0)         # aleatoric: mean of within-member sample variance
            epi = mk.var(0)                   # epistemic: variance across members
            mE = mk.mean(0); vE = ale+epi
            res['DeepEns'] = (*err_of(mE, vE, mr, vr), coverage(mE, vE, Fref), tBinf)
            for key in res:
                agg.setdefault(key, []).append(res[key])
        log(f'  {"method":<11}{"meanerr":>10}{"std err":>10}{"std corr":>9}{"cover":>8}{"cost/s":>9}')
        rowres = {}
        for key, vals in agg.items():
            arr = np.mean(vals, 0); rowres[key] = arr
            log(f'  {key:<11}{arr[0]*100:>9.3f}%{arr[1]*100:>9.3f}%{arr[2]:>9.4f}{arr[3]*100:>7.1f}%{arr[4]:>9.2f}')
        rows.append((Nx, rowres))
        save()
    # summary
    log('\n================ summary (meanerr% / std err% / coverage% / cost at target resolution) ================')
    methods = ['QMC32', 'QMC64', 'gPC(25)', 'SC-Smol', 'GP/Kriging', 'Naive', 'DeepEns', 'FrFNO']
    stat = [('meanerr%', 0, True), ('std err%', 1, True), ('std corr', 2, True),
            ('coverage%', 3, True), ('cost s', 4, False)]
    for Nx, rr in rows:
        for name, mi, pct in stat:
            line = f'{Nx+1}² {name:>8}'
            for m in methods:
                v = rr[m][mi]
                line += f'{(v*100 if pct else v):>12.3f}'
            log(line)
    log(f'\n[one-time training cost shared across resolutions] FrFNO {tA:.0f}s, deep ensemble ({K_ENS} nets) {tBtot:.0f}s, total {t_train:.0f}s')
    np.savez(os.path.join(_RDIR, 'legacy_compare_metrics.npz'),
             rows=np.array([[rr[m] for m in methods] for _, rr in rows], dtype=object),
             methods=np.array(methods))
    save()

if __name__ == '__main__':
    main()

