# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
Operator-transfer check beyond Burgers, in the paper's MAIN Dirichlet/spectral
setting (DST-I, five-point symbol, L1 implicit skeleton, internal n=Nx-1 grid).

Three reaction-diffusion mechanisms share the SAME fractional dissipative
principal part; only the zero-order term changes:
  (L)  linear variable-coeff fractional diffusion : R(u)=0
  (F)  Fisher-KPP                                   : R(u)=rho u(1-u)
  (AC) Allen-Cahn                                   : R(u)=rho (u-u^3)
PDE: ^C D_t^alpha u = -nu(x)(-Delta)^s u + R(u),  Omega=(0,1)^2, u=0 on boundary.

One-knife comparison per mechanism: FrFNO (exact frozen Mittag-Leffler propagator
+ learned residual) vs an identical FNO learning the full field; same backbone,
parameter count, optimiser, 3000-step budget. Train at Nx=16 (internal 15^2),
zero-shot super-resolve to 32,64.
"""
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, _PKG_ROOT)
from gpu_solver import dst2_torch, spectral_eigvals_torch, l1_b_torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float32
torch.manual_seed(0); np.random.seed(0)

T = 0.02
RHO = {'L': 0.0, 'F': 6.0, 'AC': 4.0}
KINDS = ['L', 'F', 'AC']
A_GRID = np.array([.7, .85, 1.], np.float32)
S_GRID = np.array([.5, .75, 1.], np.float32)
NA, NS = len(A_GRID), len(S_GRID)
NTR, NTE, STEPS, BATCH = 64, 24, 3000, 64
WIDTH, DEPTH, KMOD = 32, 4, 7
RES = [(16, 400), (32, 800), (64, 1600)]
C0 = 1.0
ROOT = _PKG_ROOT
OUT = os.path.join(ROOT, 'reaction_diffusion_result.txt')
CKDIR = ROOT
LOG = []
def log(m): print(m, flush=True); LOG.append(str(m))
def save_log(): open(OUT, 'w', encoding='utf-8').write('\n'.join(LOG))

# ---------------- Dirichlet spectral calculus (reuse paper solver) ----------------
def Qs_field(Nx, s):
    Q = spectral_eigvals_torch(Nx, DEV, DT)          # five-point symbol (n,n)
    return Q ** float(s)

def l1_b(alpha, Nt): return l1_b_torch(float(alpha), Nt, DEV, DT)

def lin_propagator(Nx, alpha, s, Nt, c=C0):
    """terminal frozen linear modal multiplier on the Dirichlet sine basis."""
    n = Nx-1
    Z = Qs_field(Nx, s).reshape(-1).double()
    b = l1_b_torch(float(alpha), Nt, DEV, torch.float64)
    dt = T/Nt; dta = dt**float(alpha)
    g = torch.zeros((Nt+1, Z.numel()), device=DEV, dtype=torch.float64)
    g[0] = 1.0; denom = b[0]+dta*c*Z
    for k in range(1, Nt+1):
        base = b[k-1]*g[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV, dtype=torch.long)
            w = b[k-j-1]-b[k-j]
            base = base + torch.einsum('j,jy->y', w, g[1:k])
        g[k] = base/denom
    return g[Nt].reshape(n, n).to(DT)

def base_field(u0, G): return dst2_torch(G*dst2_torch(u0))

def react(x, kind, rho):
    if kind == 'L':  return torch.zeros_like(x)
    if kind == 'F':  return rho*x*(1.0-x)
    if kind == 'AC': return rho*(x-x**3)
    raise ValueError(kind)

def rd_solve(u0, nu, a0, s0, kind, Nx, Nt, rho=None, n_inner=2):
    """IMEX/Picard reference: fractional diffusion implicit, reaction explicit."""
    if rho is None: rho = RHO[kind]
    B = u0.shape[0]
    Qs = Qs_field(Nx, s0)[None]
    b = l1_b(a0, Nt)[None].repeat(B, 1)
    dt = T/Nt; dta = dt**float(a0)
    c = nu.amax(dim=(1, 2), keepdim=True)
    denom = b[:, 0, None, None]+dta*c*Qs
    hist = torch.zeros((Nt+1, B, u0.shape[-1], u0.shape[-1]), device=DEV, dtype=DT)
    hist[0] = u0
    for k in range(1, Nt+1):
        base = b[:, k-1, None, None]*hist[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV)
            w = b[:, k-j-1]-b[:, k-j]
            base = base + torch.einsum('bj,jbxy->bxy', w, hist[1:k])
        x = hist[k-1]
        for _ in range(n_inner):
            Lx = dst2_torch(Qs*dst2_torch(x))
            r = base - dta*(nu-c)*Lx + dta*react(x, kind, rho)
            x = dst2_torch(dst2_torch(r)/denom)
        hist[k] = x
    return hist[Nt]

# ---------------- zero-Dirichlet random fields via double-sine synthesis ----------------
def sine_field(Nx, seed, kcut=6, decay=1.8, std=0.12, mean=0.30):
    n = Nx-1; xi = np.arange(1, Nx)/Nx
    X, Y = np.meshgrid(xi, xi, indexing='ij')
    rng = np.random.RandomState(seed); v = np.zeros((n, n), np.float64)
    for kx in range(1, kcut+1):
        for ky in range(1, kcut+1):
            amp = (kx**2+ky**2)**(-decay/2.0)
            v += amp*(rng.randn()*np.sin(kx*np.pi*X)*np.sin(ky*np.pi*Y))
    v = (v-v.mean())/(v.std()+1e-12)*std+mean
    return v.astype(np.float32)

def nu_field(Nx, seed):
    g = sine_field(Nx, 5000+seed, kcut=4, decay=1.5, std=1.0, mean=0.0)
    return (1.0+0.4*g/(np.abs(g).max()+1e-12)).astype(np.float32)

# ---------------- matched spectral backbone (same as Riesz check) ----------------
class SpecBlock(nn.Module):
    def __init__(self, ci, co, kmod):
        super().__init__(); self.k = kmod
        self.W = nn.Parameter(torch.randn(co, ci, 2*kmod+1, 2*kmod+1, dtype=torch.cfloat)*0.02)
        self.loc = nn.Conv2d(ci, co, 1); self.b = nn.Parameter(torch.zeros(co))
    def forward(self, x):
        B, C, H, Wd = x.shape
        Xl = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
        c = H//2; hw = min(self.k, H//2-1); sl = slice(c-hw, c+hw+1); ks = self.k
        Wc = self.W[:, :, ks-hw:ks+hw+1, ks-hw:ks+hw+1]
        Xs = torch.zeros(B, Wc.shape[0], H, Wd, dtype=torch.cfloat, device=x.device)
        Xs[:, :, sl, sl] = torch.einsum('bihw,oihw->bohw', Xl[:, :, sl, sl], Wc)
        g = torch.fft.ifft2(torch.fft.ifftshift(Xs, dim=(-2, -1)), norm='ortho').real
        return g+self.loc(x)+self.b[None, :, None, None]

class DualNet(nn.Module):
    def __init__(self, cin=2, width=WIDTH, depth=DEPTH, kmod=KMOD, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.inp = nn.Conv2d(cin, width, 1)
        self.blocks = nn.ModuleList([SpecBlock(width, width, kmod) for _ in range(depth)])
        self.out = nn.Conv2d(width, 1, 1)
        self.film = nn.ModuleList([nn.Sequential(nn.Linear(2, width), nn.GELU(),
                              nn.Linear(width, 2*width)) for _ in range(depth)])
        self.act = nn.GELU()
    def forward(self, x, af):
        v = self.inp(x)
        a = torch.tensor(af, device=x.device, dtype=x.dtype).reshape(1, 2).expand(x.shape[0], 2)
        for blk, fi in zip(self.blocks, self.film):
            v = self.act(blk(v)); gb = fi(a); g, bb = gb[:, :v.shape[1]], gb[:, v.shape[1]:]
            v = (1+g[:, :, None, None])*v+bb[:, :, None, None]
        return self.out(v).squeeze(1)

def relL2(p, t): return float(torch.linalg.norm(p-t)/(torch.linalg.norm(t)+1e-12)*100.0)

def build_train(Nx, Nt):
    n = Nx-1
    U0 = np.stack([sine_field(Nx, 1000+i) for i in range(NTR)])
    NU = np.stack([nu_field(Nx, 1000+i) for i in range(NTR)])
    SOL = {kd: np.zeros((NA, NS, NTR, n, n), np.float32) for kd in KINDS}
    G = {}
    for ia, a in enumerate(A_GRID):
        for is_, s in enumerate(S_GRID):
            G[(ia, is_)] = lin_propagator(Nx, a, s, Nt)
            for kd in KINDS:
                with torch.no_grad():
                    fo = rd_solve(torch.tensor(U0, device=DEV, dtype=DT),
                                  torch.tensor(NU, device=DEV, dtype=DT), a, s, kd, Nx, Nt)
                SOL[kd][ia, is_] = fo.cpu().numpy()
    return U0, NU, SOL, G

def train_one(kind, model_kind, U0, NU, SOLk, G, Nx, Nt, seed=1):
    torch.manual_seed(seed); np.random.seed(seed)
    net = DualNet(seed=seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, 5e-5)
    U0t = torch.tensor(U0, device=DEV, dtype=DT); NUt = torch.tensor(NU, device=DEV, dtype=DT)
    mean = float(SOLk.mean()); sv = float(SOLk.std()); rng = np.random.RandomState(123)
    ck = os.path.join(CKDIR, 'rd_ckpt_%s_%s.pt' % (kind, model_kind))
    t0 = time.time(); start = 0
    if os.path.exists(ck):
        d = torch.load(ck, map_location=DEV); net.load_state_dict(d['net']); opt.load_state_dict(d['opt']); start = d['step']
    for st in range(start, STEPS+1):
        ia, is_ = rng.randint(NA), rng.randint(NS); idx = rng.permutation(NTR)[:BATCH]
        u0 = U0t[idx]; nu = NUt[idx]; sol = torch.tensor(SOLk[ia, is_, idx], device=DEV, dtype=DT)
        with torch.no_grad(): base = base_field(u0, G[(ia, is_)])
        xb = torch.stack([u0, nu], 1); out = net(xb, (A_GRID[ia], S_GRID[is_]))
        target = (sol-base)/sv if model_kind == 'A' else (sol-mean)/sv
        loss = F.mse_loss(out, target)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if st % 500 == 0: torch.save({'step': st, 'net': net.state_dict(), 'opt': opt.state_dict()}, ck)
    if os.path.exists(ck): os.remove(ck)
    net.eval(); return net, time.time()-t0, mean, sv

_GCACHE = {}
@torch.no_grad()
def evaluate(net, model_kind, kind, Nx, Nt, Ue, Ne, a_te, s_te, mean, sv):
    # orders lie on the 3x3 grid -> group by order and solve/run the net as a batch
    errs = []
    keys = sorted(set(zip(np.round(a_te, 4).tolist(), np.round(s_te, 4).tolist())))
    for (a, s) in keys:
        idx = [b for b in range(len(Ue)) if abs(a_te[b]-a) < 1e-6 and abs(s_te[b]-s) < 1e-6]
        u0 = torch.tensor(Ue[idx], device=DEV, dtype=DT); nu = torch.tensor(Ne[idx], device=DEV, dtype=DT)
        sol = rd_solve(u0, nu, a, s, kind, Nx, Nt)
        out = net(torch.stack([u0, nu], 1), (a, s))
        if model_kind == 'A':
            key = (Nx, round(a, 4), round(s, 4), Nt)
            if key not in _GCACHE: _GCACHE[key] = lin_propagator(Nx, a, s, Nt)
            pred = base_field(u0, _GCACHE[key])+sv*out
        else: pred = mean+sv*out
        for j in range(len(idx)): errs.append(relL2(pred[j], sol[j]))
    return float(np.mean(errs))

def main():
    tg = time.time()
    log('Reaction-diffusion operator-transfer (Dirichlet DST, T=%g; mechanisms %s)' % (T, KINDS))
    log('train Nx=16 internal 15^2, NTR=%d steps=%d order %dx%d, NTE=%d, res=%s' % (NTR, STEPS, NA, NS, NTE, RES))
    Nx0, Nt0 = RES[0]
    U0, NU, SOL, G16 = build_train(Nx0, Nt0)
    # how much does each reaction change the purely-linear solution (regime check)
    for kd in KINDS:
        lin = SOL['L'][1, 1]; nonl = SOL[kd][1, 1]
        chg = float(np.linalg.norm(nonl-lin)/(np.linalg.norm(lin)+1e-12)*100)
        log('regime: %s changes the linear solution by %.1f%% (at alpha=.85,s=.75)' % (kd, chg))
    save_log()
    rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=NTE)
    # test orders sampled on the 3x3 order grid (enables batched reference solve)
    ia_te = rng.randint(NA, size=NTE); is_te = rng.randint(NS, size=NTE)
    a_te = A_GRID[ia_te]; s_te = S_GRID[is_te]
    results = {}
    for kd in KINDS:
        results[kd] = {}
        for mk, tag, sd in [('A', 'FrFNO', 1), ('F', 'FNO', 2)]:
            net, tt, mean, sv = train_one(kd, mk, U0, NU, SOL[kd], G16, Nx0, Nt0, seed=sd)
            row = {'train_s': tt}
            for Nx, Nt in RES:
                Ue = np.stack([sine_field(Nx, int(sd2)) for sd2 in seeds]); Ne = np.stack([nu_field(Nx, int(sd2)) for sd2 in seeds])
                row['nx%d' % Nx] = evaluate(net, mk, kd, Nx, Nt, Ue, Ne, a_te, s_te, mean, sv)
            results[kd][tag] = row
            log('[%s/%s] 16:%.3f%% 32:%.3f%% 64:%.3f%% (%.1fs)' % (
                kd, tag, row['nx16'], row['nx32'], row['nx64'], tt)); save_log()
    log('\n===== reaction-diffusion summary: relL2 (%) =====')
    log('%-3s %-7s %9s %9s %9s %10s' % ('mech', 'model', '16', '32(ZS)', '64(ZS)', 'train_s'))
    for kd in KINDS:
        for tag in ['FrFNO', 'FNO']:
            r = results[kd][tag]
            log('%-3s %-7s %8.3f%% %8.3f%% %8.3f%% %9.1f' % (kd, tag, r['nx16'], r['nx32'], r['nx64'], r['train_s']))
    np.savez(os.path.join(CKDIR, 'reaction_diffusion_metrics.npz'), results=json.dumps(results))
    log('total %.1fs' % (time.time()-tg)); save_log()

if __name__ == '__main__': main()
