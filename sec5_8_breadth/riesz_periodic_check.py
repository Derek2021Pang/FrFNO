# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
Periodic (torus) Riesz fractional-Laplacian robustness check for FrFNO.

Purpose (rebuttal experiment): the paper uses the spectral/Dirichlet fractional
Laplacian, diagonal on the sine (DST) basis. On the periodic torus the integral
(Riesz) fractional Laplacian is *identical* to the spectral one as an operator and
is diagonal on the Fourier basis with symbol |k|^{2s}. This script replaces DST by
FFT and the five-point symbol by |k|^{2s}, and reruns the one-knife comparison:
  FrFNO = exact frozen linear Mittag-Leffler propagator (FFT multiplier) + learned residual
  FNO   = identical backbone learning the FULL field (base = 0)
Everything else (data, backbone, parameter count, optimiser, budget) is matched.

Equation on T^2 = [0,1)^2 (periodic):
  ^C D_t^alpha u = -nu(x) (-Delta_R)^s u - beta u d_x u,
with (-Delta_R)^s the periodic Riesz fractional Laplacian, FFT symbol (kx^2+ky^2)^s.

Trains at n=16, zero-shot super-resolves to 32,64. Writes incremental checkpoint
and a text/npz summary; supports resume.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float32
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------- problem / budget ----------------
BETA = 3.0
T = 0.015
NT_TRAIN = 400                 # reference L1 steps at training resolution
NTR = 64                       # training random fields
NTE = 24                       # test fields
A_GRID = np.array([0.7, 0.85, 1.0], np.float32)
S_GRID = np.array([0.5, 0.75, 1.0], np.float32)
NA, NS = len(A_GRID), len(S_GRID)
STEPS = 3000
BATCH = 64
WIDTH, DEPTH, KMOD = 32, 4, 7   # 2K+1=15 <= train grid 16: all spectral weights are trained
RES = [(16, 400), (32, 800), (64, 1600)]   # (n, Nt) train then zero-shot super-res
C0 = 1.0                       # frozen diffusivity for the analytic propagator
ROOT = _PKG_ROOT
OUT_TXT = os.path.join(ROOT, 'riesz_periodic_result.txt')
CKPT = os.path.join(ROOT, 'riesz_periodic_ckpt.pt')
LOG = []
def log(m): print(m, flush=True); LOG.append(str(m))
def save_log(): open(OUT_TXT, 'w', encoding='utf-8').write('\n'.join(LOG))

# ---------------- periodic spectral calculus ----------------
def riesz_symbol(n, s, device=DEV, dtype=DT):
    """(kx^2+ky^2)^s on the periodic n-grid (length 2pi = domain [0,1))."""
    k = 2*np.pi*np.fft.fftfreq(n, d=1.0/n)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    K2 = (kx**2 + ky**2).astype(np.float64)
    K2[0, 0] = 0.0
    return torch.tensor(K2**float(s), device=device, dtype=dtype)

def fft2(u):  return torch.fft.fft2(u, norm='ortho')
def ifft2(U): return torch.fft.ifft2(U, norm='ortho').real

def ddx_upwind_periodic(u, n):
    dx = 1.0/n
    back = (u - torch.roll(u, 1, dims=-1))/dx
    fwd  = (torch.roll(u, -1, dims=-1) - u)/dx
    return torch.where(u >= 0, back, fwd)

def l1_weights(alpha, Nt):
    j = torch.arange(Nt, device=DEV, dtype=DT)
    return (j+1.0)**(1.0-alpha) - torch.where(j == 0, torch.zeros_like(j), j**(1.0-alpha))

def linear_propagator(n, alpha, s, Nt, c=C0):
    """Terminal frozen-coefficient modal multiplier g^N(z), z=|k|^{2s} (L1 scheme)."""
    Z = riesz_symbol(n, s).reshape(-1).double()
    b = l1_weights(float(alpha), Nt).double()
    dt = T/Nt
    dta = dt**float(alpha)
    g = torch.zeros((Nt+1, Z.numel()), device=DEV, dtype=torch.float64)
    g[0] = 1.0
    denom = 1.0 + dta*c*Z
    for k in range(1, Nt+1):
        base = b[k-1]*g[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV, dtype=torch.long)
            w = b[k-j-1] - b[k-j]
            base = base + torch.einsum('j,jy->y', w, g[1:k])
        g[k] = base/denom
    return g[Nt].reshape(n, n).to(DT)

def base_field(u0, G):
    return ifft2(G*fft2(u0))

def riesz_nl_solve(U0, NU, alpha, s, Nt, n, beta=BETA, n_inner=2):
    """Periodic IMEX/Picard reference solver, batched. U0,NU:(B,n,n)."""
    B = U0.shape[0]
    a0 = float(np.asarray(alpha).reshape(-1)[0])      # one order per batched call
    s0 = float(np.asarray(s).reshape(-1)[0])
    Z = riesz_symbol(n, s0)[None]                     # (1,n,n)
    b = l1_weights(a0, Nt)[None].repeat(B, 1)         # (B,Nt)
    dt = T/Nt
    dta = dt**a0                                      # scalar (same order in batch)
    c = NU.amax(dim=(1, 2), keepdim=True)
    denom = 1.0 + dta*c*Z
    hist = torch.zeros((Nt+1, B, n, n), device=DEV, dtype=DT)
    hist[0] = U0
    for k in range(1, Nt+1):
        base = b[:, k-1, None, None]*hist[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV)
            w = b[:, k-j-1] - b[:, k-j]
            base = base + torch.einsum('bj,jbxy->bxy', w, hist[1:k])
        adv = hist[k-1]*ddx_upwind_periodic(hist[k-1], n)
        r = base - dta*beta*adv
        x = hist[k-1]
        for _ in range(n_inner):
            Lx = ifft2(Z*fft2(x))
            rr = r - dta*(NU-c)*Lx
            x = ifft2(fft2(rr)/denom)
        hist[k] = x
    return hist[Nt]

# ---------------- random fields (periodic, smooth) ----------------
def periodic_field(n, seed, kcut, decay, std=1.0, mean=0.0):
    rng = np.random.RandomState(seed)
    k = np.fft.fftfreq(n, d=1.0/n)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    km = np.maximum(np.abs(kx), np.abs(ky))
    amp = np.where(km <= kcut, (1.0+km)**(-decay), 0.0)
    c = (rng.randn(n, n)+1j*rng.randn(n, n))*amp
    v = np.fft.ifft2(c, norm='ortho').real
    v = (v-v.mean())/(v.std()+1e-12)*std + mean
    return v.astype(np.float32)

def make_u0(n, seed):  return periodic_field(n, seed, 8, 1.8, std=0.22, mean=0.005)
def make_nu(n, seed):
    g = periodic_field(n, 10000+seed, 4, 1.5, std=1.0, mean=0.0)
    return (1.0 + 0.4*g/(np.abs(g).max()+1e-12)).astype(np.float32)

# ---------------- matched spectral backbone ----------------
class SpecBlock(nn.Module):
    def __init__(self, ci, co, kmod):
        super().__init__()
        self.k = kmod
        self.W = nn.Parameter(torch.randn(co, ci, 2*kmod+1, 2*kmod+1, dtype=torch.cfloat)*0.02)
        self.loc = nn.Conv2d(ci, co, 1)
        self.b = nn.Parameter(torch.zeros(co))
    def forward(self, x):
        B, C, H, W = x.shape
        X = torch.fft.fft2(x, norm='ortho')
        Xl = torch.fft.fftshift(X, dim=(-2, -1))
        c = H//2
        hw = min(self.k, H//2-1)          # adaptive low-mode window (fixed K when grid allows)
        sl = slice(c-hw, c+hw+1)
        ks = self.k
        Wc = self.W[:, :, ks-hw:ks+hw+1, ks-hw:ks+hw+1]
        Xs = torch.zeros(B, Wc.shape[0], H, W, dtype=torch.cfloat, device=x.device)
        Xwin = Xl[:, :, sl, sl]
        out = torch.einsum('bihw,oihw->bohw', Xwin, Wc)
        Xs[:, :, sl, sl] = out
        Xs = torch.fft.ifftshift(Xs, dim=(-2, -1))
        g = torch.fft.ifft2(Xs, norm='ortho').real
        return g + self.loc(x) + self.b[None, :, None, None]

class DualNet(nn.Module):
    def __init__(self, cin=2, width=WIDTH, depth=DEPTH, kmod=KMOD, seed=0):
        super().__init__()
        torch.manual_seed(seed)
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
            v = self.act(blk(v))
            gb = fi(a)
            g, bb = gb[:, :v.shape[1]], gb[:, v.shape[1]:]
            v = (1+g[:, :, None, None])*v + bb[:, :, None, None]
        return self.out(v).squeeze(1)

# ---------------- data ----------------
def build_dataset(n, Nt, nfield, seed0):
    U0 = np.stack([make_u0(n, seed0+i) for i in range(nfield)])
    NU = np.stack([make_nu(n, seed0+i) for i in range(nfield)])
    SOL = np.zeros((NA, NS, nfield, n, n), np.float32)
    Gtab = {}
    for ia, a in enumerate(A_GRID):
        for is_, s in enumerate(S_GRID):
            Gtab[(ia, is_)] = linear_propagator(n, a, s, Nt)
            with torch.no_grad():
                fo = riesz_nl_solve(torch.tensor(U0, device=DEV, dtype=DT),
                                    torch.tensor(NU, device=DEV, dtype=DT),
                                    np.full(nfield, a, np.float32),
                                    np.full(nfield, s, np.float32), Nt, n)
            SOL[ia, is_] = fo.cpu().numpy()
    return U0, NU, SOL, Gtab

def relL2(p, t):
    return float(torch.linalg.norm(p-t)/ (torch.linalg.norm(t)+1e-12)*100.0)

def train_model(kind, U0, NU, SOL, G16, steps=STEPS, seed=1):
    """kind='A' FrFNO (base+residual); kind='F' FNO (full field)."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = DualNet(seed=seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, 5e-5)
    U0t = torch.tensor(U0, device=DEV, dtype=DT)
    NUt = torch.tensor(NU, device=DEV, dtype=DT)
    mean = float(SOL.mean()); sv = float(SOL.std())
    n = U0.shape[-1]
    # inputs are scale-consistent across resolutions (u0, nu only); order enters via FiLM
    rng = np.random.RandomState(123)
    t0 = time.time()
    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=DEV)
        if ck.get('kind') == kind:
            net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt']); start = ck['step']
    for st in range(start, steps+1):
        ia, is_ = rng.randint(NA), rng.randint(NS)
        idx = rng.permutation(NTR)[:BATCH]
        u0 = U0t[idx]; nu = NUt[idx]
        sol = torch.tensor(SOL[ia, is_, idx], device=DEV, dtype=DT)
        G = G16[(ia, is_)]
        with torch.no_grad(): base = base_field(u0, G)
        xb = torch.stack([u0, nu], 1)
        pred = net(xb, (A_GRID[ia], S_GRID[is_]))
        if kind == 'A':
            target = (sol-base)/sv
            loss = F.mse_loss(pred, target)
        else:
            loss = F.mse_loss(pred, (sol-mean)/sv)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if st % 500 == 0:
            torch.save({'kind': kind, 'step': st, 'net': net.state_dict(),
                        'opt': opt.state_dict()}, CKPT)
    if os.path.exists(CKPT): os.remove(CKPT)
    net.eval()
    return net, time.time()-t0, mean, sv

@torch.no_grad()
def evaluate(net, kind, n, Nt, U0, NU, a_arr, s_arr, mean, sv):
    # reference, base and network all use the SAME continuous test order per sample
    errs = []
    for b in range(len(U0)):
        a, s = float(a_arr[b]), float(s_arr[b])
        u0 = torch.tensor(U0[b:b+1], device=DEV, dtype=DT)
        nu = torch.tensor(NU[b:b+1], device=DEV, dtype=DT)
        sol = riesz_nl_solve(u0, nu, np.full(1, a, np.float32),
                             np.full(1, s, np.float32), Nt, n)[0]
        xb = torch.stack([u0, nu], 1)
        out = net(xb, (a, s))
        if kind == 'A':
            G = linear_propagator(n, a, s, Nt)
            pred = base_field(u0, G)[0] + sv*out[0]
        else:
            pred = mean + sv*out[0]
        errs.append(relL2(pred, sol))
    return float(np.mean(errs))

def main():
    t_glob = time.time()
    log('Periodic Riesz fractional-Laplacian check (FFT symbol |k|^{2s}), beta=%g T=%g' % (BETA, T))
    log('train n=16 NTR=%d steps=%d order grid %dx%d; test NTE=%d; res=%s'
        % (NTR, STEPS, NA, NS, NTE, RES))
    U0tr, NUtr, SOLtr, G16 = build_dataset(16, NT_TRAIN, NTR, seed0=1000)
    log('train reference built: SOL shape %s' % (SOLtr.shape,)); save_log()

    results = {}
    for kind, tag, seed in [('A', 'FrFNO', 1), ('F', 'FNO', 2)]:
        net, tt, mean, sv = train_model(kind, U0tr, NUtr, SOLtr, G16, seed=seed)
        results[tag] = {'train_s': tt}
        log('[%s] trained in %.1fs' % (tag, tt)); save_log()
        rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=NTE)
        a_te = rng.uniform(A_GRID.min(), A_GRID.max(), NTE).astype(np.float32)
        s_te = rng.uniform(S_GRID.min(), S_GRID.max(), NTE).astype(np.float32)
        for n, Nt in RES:
            Ue = np.stack([make_u0(n, int(sd)) for sd in seeds])
            Ne = np.stack([make_nu(n, int(sd)) for sd in seeds])
            err = evaluate(net, kind, n, Nt, Ue, Ne, a_te, s_te, mean, sv)
            results[tag]['n%d' % n] = err
            log('  [%s] n=%d  relL2 = %.3f%%' % (tag, n, err)); save_log()
    log('\n===== periodic Riesz summary: relL2 (%) =====')
    log('%-8s %10s %10s %10s %12s' % ('model', '16(train)', '32(ZS)', '64(ZS)', 'train_s'))
    for tag in ['FrFNO', 'FNO']:
        r = results[tag]
        log('%-8s %9.3f%% %9.3f%% %9.3f%% %11.1f' % (tag, r['n16'], r['n32'], r['n64'], r['train_s']))
    np.savez(os.path.join(ROOT, 'riesz_periodic_metrics.npz'),
             results=json.dumps(results))
    log('total %.1fs' % (time.time()-t_glob)); save_log()

if __name__ == '__main__':
    main()
