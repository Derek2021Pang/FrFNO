# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
Extension to the incompressible fractional Navier--Stokes equations via the
2D vorticity--streamfunction formulation on the periodic torus T^2.

  ^C D_t^a w = -nu (-Delta_R)^s w - u.dx w - v.dy w,
  -Delta psi = w,   (u,v) = (dy psi, -dx psi)   (Biot--Savart, divergence free).

The vorticity w is a SCALAR; its linear principal part -nu(-Delta)^s w is exactly
the scalar fractional diffusion already covered by the frozen Mittag-Leffler/
L1 propagator (FFT multiplier). The genuinely new, nonlinear and nonlocal piece
is the advection u.grad w, where the velocity is recovered from w at every step
by the spectral Biot--Savart inversion psi_hat = w_hat/|k|^2; it is learned by
the residual. Incompressibility is automatic (u,v) = grad^perp psi, so no
divergence-free basis or pressure is required. Constant (per-sample) viscosity
nu in {0.8,1.0,1.2} makes the propagator exact for the linear part; the residual
learns only the nonlinear vortex advection.

Matched comparison on the same spectral backbone:
  FrFNO = exact frozen linear propagator + learned residual;
  FNO   = identical backbone learning the full vorticity field.
Trains at n=16, zero-shot super-resolves 32,64. Incremental checkpoint + resume.
"""
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; DT = torch.float32
torch.manual_seed(0); np.random.seed(0)
T = 0.15
A_GRID = np.array([.7, .85, 1.], np.float32); S_GRID = np.array([.6, .8, 1.], np.float32)
VNU = np.array([.10, .15, .20], np.float32)         # constant per-sample viscosity grid
NA, NS, NV = len(A_GRID), len(S_GRID), len(VNU)
NTR, NTE, STEPS, BATCH = 64, 24, 3000, 64
WIDTH, DEPTH, KMOD = 32, 4, 7
RES = [(16, 400), (32, 800), (64, 1600)]
ROOT = _PKG_ROOT
CKDIR = ROOT; OUT = os.path.join(CKDIR, 'ns_vorticity_result.txt')
CKPT = os.path.join(CKDIR, 'ns_ckpt.pt')
LOG = []
def log(m): print(m, flush=True); LOG.append(str(m))
def save_log(): open(OUT, 'w', encoding='utf-8').write('\n'.join(LOG))

# ---------- periodic spectral calculus ----------
_WAVE = {}
def waves(n):
    if n not in _WAVE:
        k = 2*np.pi*np.fft.fftfreq(n, d=1.0/n)
        kx, ky = np.meshgrid(k, k, indexing='ij')
        K2 = (kx**2+ky**2); k2inv = np.zeros_like(K2); nz = K2 > 0
        k2inv[nz] = 1.0/K2[nz]
        _WAVE[n] = (torch.tensor(kx, device=DEV, dtype=DT), torch.tensor(ky, device=DEV, dtype=DT),
                    torch.tensor(k2inv, device=DEV, dtype=DT))
    return _WAVE[n]

def fft2(u): return torch.fft.fft2(u, norm='ortho')
def ifft2(U): return torch.fft.ifft2(U, norm='ortho').real
def riesz_symbol(n, s):
    k = 2*np.pi*np.fft.fftfreq(n, d=1.0/n); kx, ky = np.meshgrid(k, k, indexing='ij')
    K2 = (kx**2+ky**2).astype(np.float64); K2[0, 0] = 0.0
    return torch.tensor(K2**float(s), device=DEV, dtype=DT)

def biot_savart(w):
    n = w.shape[-1]; kx, ky, k2inv = waves(n)
    Wh = fft2(w); psi = Wh*k2inv[None]
    u = ifft2(1j*ky[None]*psi); v = ifft2(-1j*kx[None]*psi)
    return u, v

def grad_up(vel, q, n, axis):
    dx = 1.0/n; dim = -1 if axis == 'x' else -2
    back = (q-torch.roll(q, 1, dims=dim))/dx; fwd = (torch.roll(q, -1, dims=dim)-q)/dx
    return torch.where(vel >= 0, back, fwd)

def vort_advection(w):
    n = w.shape[-1]; u, v = biot_savart(w)
    return u*grad_up(u, w, n, 'x')+v*grad_up(v, w, n, 'y')

def l1_weights(alpha, Nt):
    j = torch.arange(Nt, device=DEV, dtype=DT)
    return (j+1.)**(1.-alpha)-torch.where(j == 0, torch.zeros_like(j), j**(1.-alpha))

def linear_propagator(n, alpha, s, Nt, nu):
    Z = riesz_symbol(n, s).reshape(-1).double(); b = l1_weights(float(alpha), Nt).double()
    dt = T/Nt; dta = dt**float(alpha); g = torch.zeros((Nt+1, Z.numel()), device=DEV, dtype=torch.float64)
    g[0] = 1.; denom = 1.+dta*float(nu)*Z
    for k in range(1, Nt+1):
        base = b[k-1]*g[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV); base = base+torch.einsum('j,jy->y', b[k-j-1]-b[k-j], g[1:k])
        g[k] = base/denom
    return g[Nt].reshape(n, n).to(DT)

def base_field(w0, G): return ifft2(G*fft2(w0))

def ns_solve(w0, nu, a0, s0, Nt, n, adv_on=True):
    B = w0.shape[0]; Z = riesz_symbol(n, s0)[None]; b = l1_weights(a0, Nt)[None].repeat(B, 1)
    dt = T/Nt; dta = dt**float(a0); denom = 1.+dta*float(nu)*Z
    hist = torch.zeros((Nt+1, B, n, n), device=DEV, dtype=DT); hist[0] = w0
    for k in range(1, Nt+1):
        base = b[:, k-1, None, None]*hist[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV); w = b[:, k-j-1]-b[:, k-j]
            base = base+torch.einsum('bj,jbxy->bxy', w, hist[1:k])
        adv = vort_advection(hist[k-1]) if adv_on else 0.
        r = base-dta*adv
        hist[k] = ifft2(fft2(r)/denom)
    return hist[Nt]

def periodic_field(n, seed, kcut, decay, std=1., mean=0.):
    rng = np.random.RandomState(seed); k = np.fft.fftfreq(n, d=1./n)
    kx, ky = np.meshgrid(k, k, indexing='ij'); km = np.maximum(np.abs(kx), np.abs(ky))
    amp = np.where(km <= kcut, (1.+km)**(-decay), 0.)
    c = (rng.randn(n, n)+1j*rng.randn(n, n))*amp
    v = np.fft.ifft2(c, norm='ortho').real
    return ((v-v.mean())/(v.std()+1e-12)*std+mean).astype(np.float32)
def make_w0(n, seed): return periodic_field(n, seed, 8, 1.8, std=1.20, mean=0.)

class SpecBlock(nn.Module):
    def __init__(self, ci, co, kmod):
        super().__init__(); self.k = kmod
        self.W = nn.Parameter(torch.randn(co, ci, 2*kmod+1, 2*kmod+1, dtype=torch.cfloat)*.02)
        self.loc = nn.Conv2d(ci, co, 1); self.b = nn.Parameter(torch.zeros(co))
    def forward(self, x):
        B, C, H, Wd = x.shape; Xl = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
        c = H//2; hw = min(self.k, H//2-1); sl = slice(c-hw, c+hw+1); ks = self.k
        Wc = self.W[:, :, ks-hw:ks+hw+1, ks-hw:ks+hw+1]
        Xs = torch.zeros(B, Wc.shape[0], H, Wd, dtype=torch.cfloat, device=x.device)
        Xs[:, :, sl, sl] = torch.einsum('bihw,oihw->bohw', Xl[:, :, sl, sl], Wc)
        return torch.fft.ifft2(torch.fft.ifftshift(Xs, dim=(-2, -1)), norm='ortho').real+self.loc(x)+self.b[None, :, None, None]

class Net(nn.Module):
    def __init__(self, cin=2, width=WIDTH, depth=DEPTH, kmod=KMOD, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.inp = nn.Conv2d(cin, width, 1)
        self.blocks = nn.ModuleList([SpecBlock(width, width, kmod) for _ in range(depth)])
        self.out = nn.Conv2d(width, 1, 1)
        self.film = nn.ModuleList([nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, 2*width)) for _ in range(depth)])
        self.act = nn.GELU()
    def forward(self, x, af):
        v = self.inp(x); a = torch.tensor(af, device=x.device, dtype=x.dtype).reshape(1, 2).expand(x.shape[0], 2)
        for blk, fi in zip(self.blocks, self.film):
            v = self.act(blk(v)); gb = fi(a); g, bb = gb[:, :v.shape[1]], gb[:, v.shape[1]:]
            v = (1+g[:, :, None, None])*v+bb[:, :, None, None]
        return self.out(v).squeeze(1)

def relL2(p, t): return float(torch.linalg.norm(p-t)/(torch.linalg.norm(t)+1e-12)*100.)

def build(n, Nt):
    W0 = np.stack([make_w0(n, 1000+i) for i in range(NTR)])
    SOL = np.zeros((NA, NS, NV, NTR, n, n), np.float32); LIN = np.zeros_like(SOL); Gt = {}
    w0t = torch.tensor(W0, device=DEV, dtype=DT)
    for ia, a in enumerate(A_GRID):
        for is_, s in enumerate(S_GRID):
            for iv, nu in enumerate(VNU):
                Gt[(ia, is_, iv)] = linear_propagator(n, a, s, Nt, nu)
                with torch.no_grad():
                    SOL[ia, is_, iv] = ns_solve(w0t, nu, a, s, Nt, n, True).cpu().numpy()
                    LIN[ia, is_, iv] = ns_solve(w0t, nu, a, s, Nt, n, False).cpu().numpy()
    return W0, SOL, LIN, Gt

def train_one(kind, W0, SOL, G16, seed=1):
    torch.manual_seed(seed); np.random.seed(seed); net = Net(seed=seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, 5e-5)
    w0t = torch.tensor(W0, device=DEV, dtype=DT); mean = float(SOL.mean()); sv = float(SOL.std())
    rng = np.random.RandomState(123); t0 = time.time(); start = 0
    if os.path.exists(CKPT):
        d = torch.load(CKPT, map_location=DEV)
        if d.get('kind') == kind: net.load_state_dict(d['net']); opt.load_state_dict(d['opt']); start = d['step']
    for st in range(start, STEPS+1):
        ia, is_, iv = rng.randint(NA), rng.randint(NS), rng.randint(NV); idx = rng.permutation(NTR)[:BATCH]
        w0 = w0t[idx]; sol = torch.tensor(SOL[ia, is_, iv, idx], device=DEV, dtype=DT)
        nu = VNU[iv]
        with torch.no_grad(): bf = base_field(w0, G16[(ia, is_, iv)])
        nf = torch.full_like(w0, float(nu)); xb = torch.stack([w0, nf], 1); out = net(xb, (A_GRID[ia], S_GRID[is_]))
        target = (sol-bf)/sv if kind == 'A' else (sol-mean)/sv
        loss = F.mse_loss(out, target); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if st % 500 == 0: torch.save({'kind': kind, 'step': st, 'net': net.state_dict(), 'opt': opt.state_dict()}, CKPT)
    if os.path.exists(CKPT): os.remove(CKPT)
    net.eval(); return net, time.time()-t0, mean, sv

_GC = {}
@torch.no_grad()
def evaluate(net, kind, n, Nt, We, a_te, s_te, nu_te, mean, sv):
    ew, em = [], []
    for (a, s, nu) in sorted(set(zip(np.round(a_te, 4).tolist(), np.round(s_te, 4).tolist(), np.round(nu_te, 4).tolist()))):
        idx = [b for b in range(len(We)) if abs(a_te[b]-a) < 1e-6 and abs(s_te[b]-s) < 1e-6 and abs(nu_te[b]-nu) < 1e-6]
        w0 = torch.tensor(We[idx], device=DEV, dtype=DT); sol = ns_solve(w0, nu, a, s, Nt, n)
        nf = torch.full_like(w0, float(nu)); out = net(torch.stack([w0, nf], 1), (a, s))
        key = (n, round(a, 4), round(s, 4), round(float(nu), 4), Nt)
        if key not in _GC: _GC[key] = linear_propagator(n, a, s, Nt, nu)
        pred = base_field(w0, _GC[key])+sv*out if kind == 'A' else mean+sv*out
        us, vs = biot_savart(sol); up, vp = biot_savart(pred)
        ms = torch.sqrt(us**2+vs**2); mp = torch.sqrt(up**2+vp**2)
        for j in range(len(idx)): ew.append(relL2(pred[j], sol[j])); em.append(relL2(mp[j], ms[j]))
    return float(np.mean(ew)), float(np.mean(em))

def main():
    tg = time.time()
    log('2D incompressible fractional NS (vorticity-streamfunction), periodic FFT, T=%g' % T)
    log('orders %dx%d, viscosity grid %s, NTR=%d, steps=%d' % (NA, NS, VNU.tolist(), NTR, STEPS))
    # quick physical sanity: divergence-free + linear consistency
    with torch.no_grad():
        w = torch.tensor(np.stack([make_w0(16, 1), make_w0(16, 2)]), device=DEV, dtype=DT)
        u, v = biot_savart(w); kx, ky, _ = waves(16)
        div = ifft2(1j*kx[None]*fft2(u)+1j*ky[None]*fft2(v)); log('Biot-Savart divergence-free residual %.2e' % float(div.abs().max()))
        G = linear_propagator(16, .85, .8, 120, 1.0); lin = ns_solve(w, 1., .85, .8, 120, 16, False)
        log('linear (advection-off) propagator consistency %.2e' % float(torch.linalg.norm(lin-base_field(w, G))/torch.linalg.norm(lin)))
    W0, SOL, LIN, G16 = build(16, 400)
    nl = float(np.linalg.norm(SOL-LIN)/(np.linalg.norm(LIN)+1e-12)*100)
    log('regime: vortex advection changes the linear solution by %.1f%%; train SOL built' % nl); save_log()
    rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=NTE)
    a_te = A_GRID[rng.randint(NA, size=NTE)]; s_te = S_GRID[rng.randint(NS, size=NTE)]; nu_te = VNU[rng.randint(NV, size=NTE)]
    res = {}
    for kind, tag, sd in [('A', 'FrFNO', 1), ('F', 'FNO', 2)]:
        net, tt, mean, sv = train_one(kind, W0, SOL, G16, seed=sd); row = {'train_s': tt}
        for n, Nt in RES:
            We = np.stack([make_w0(n, int(x)) for x in seeds])
            row['n%d' % n] = evaluate(net, kind, n, Nt, We, a_te, s_te, nu_te, mean, sv)
        res[tag] = row
        log('[%s] vort %.3f/%.3f/%.3f |vel| %.3f/%.3f/%.3f %.1fs' % (
            tag, row['n16'][0], row['n32'][0], row['n64'][0], row['n16'][1], row['n32'][1], row['n64'][1], tt)); save_log()
    log('\n===== fractional incompressible NS: vorticity relL2 (%) / velocity-magnitude relL2 (%) =====')
    log('%-7s %16s %16s %16s %9s' % ('model', '16(train)', '32(ZS)', '64(ZS)', 'train_s'))
    for tag in res:
        r = res[tag]; log('%-7s %7.3f/%6.3f %7.3f/%6.3f %7.3f/%6.3f %8.1f' % (
            tag, r['n16'][0], r['n16'][1], r['n32'][0], r['n32'][1], r['n64'][0], r['n64'][1], r['train_s']))
    np.savez(os.path.join(CKDIR, 'ns_vorticity_metrics.npz'), results=json.dumps(res))
    log('total %.1fs' % (time.time()-tg)); save_log()

if __name__ == '__main__': main()
