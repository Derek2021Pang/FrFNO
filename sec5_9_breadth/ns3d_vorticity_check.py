# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
Small-scale 3D proof-of-concept: incompressible fractional Navier--Stokes in
VECTOR vorticity form on the periodic 3-torus.

  ^C D_t^a w = -nu (-Delta_R)^s w + (w.grad)u - (u.grad)w,   w = curl u, div u = 0.

Compared with the 2D scalar case the vorticity is now a 3-vector and a genuinely
new nonlinear term appears: vortex stretching (w.grad)u. The velocity is
recovered by the 3D spectral Biot--Savart inversion  u_hat = i (k x w_hat)/|k|^2
(the transverse projection makes u divergence-free; initial vorticity is built
as the curl of a smooth vector potential so that div w = 0 identically). The
linear principal part is still componentwise scalar fractional diffusion, so the
SAME scalar Mittag--Leffler/L1 propagator acts on all three components and only
the (now vector, stretching-containing) nonlinearity is learned.

Deliberately small (concept check): train n=12, zero-shot 16,24; 2x2 order grid;
fixed viscosity; FrFNO vs matched FNO.
"""
import os, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; DT = torch.float32
torch.manual_seed(0); np.random.seed(0)
T = 0.10; NU = 0.15
A_GRID = np.array([.85, 1.], np.float32); S_GRID = np.array([.75, 1.], np.float32)
NA, NS = len(A_GRID), len(S_GRID)
NTR, NTE, STEPS, BATCH = 32, 8, 1500, 32
WIDTH, DEPTH, KMOD = 24, 3, 5
RES = [(12, 200), (16, 300), (24, 460)]
ROOT = _PKG_ROOT
CKDIR = ROOT; OUT = os.path.join(CKDIR, 'ns3d_result.txt')
CKPT = os.path.join(CKDIR, 'ns3d_ckpt.pt')
LOG = []
def log(m): print(m, flush=True); LOG.append(str(m))
def save_log(): open(OUT, 'w', encoding='utf-8').write('\n'.join(LOG))

_W = {}
def waves(n):
    if n not in _W:
        k = 2*np.pi*np.fft.fftfreq(n, d=1./n)
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        K2 = (kx**2+ky**2+kz**2).astype(np.float64); k2i = np.zeros_like(K2); nz = K2 > 0; k2i[nz] = 1./K2[nz]
        t = lambda a: torch.tensor(a, device=DEV, dtype=DT)
        _W[n] = (t(kx), t(ky), t(kz), t(k2i))
    return _W[n]

def fftn(u): return torch.fft.fftn(u, dim=(-3, -2, -1), norm='ortho')
def ifftn(U): return torch.fft.ifftn(U, dim=(-3, -2, -1), norm='ortho').real
def symb(n, s):
    k = 2*np.pi*np.fft.fftfreq(n, d=1./n); kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    K2 = (kx**2+ky**2+kz**2).astype(np.float64); K2[0, 0, 0] = 0.
    return torch.tensor(K2**float(s), device=DEV, dtype=DT)

def biot_savart(w):  # w:(B,3,n,n,n) -> u same; u_hat = i (k x w_hat)/|k|^2
    n = w.shape[-1]; kx, ky, kz, k2i = waves(n); Wh = fftn(w)
    cx = ky[None]*Wh[:, 2]-kz[None]*Wh[:, 1]
    cy = kz[None]*Wh[:, 0]-kx[None]*Wh[:, 2]
    cz = kx[None]*Wh[:, 1]-ky[None]*Wh[:, 0]
    return torch.stack([ifftn(1j*cx*k2i[None]), ifftn(1j*cy*k2i[None]), ifftn(1j*cz*k2i[None])], 1)

def gup(vel, q, axis):
    dx = 1./q.shape[-1]; dim = {'x': -1, 'y': -2, 'z': -3}[axis]
    back = (q-torch.roll(q, 1, dims=dim))/dx; fwd = (torch.roll(q, -1, dims=dim)-q)/dx
    return torch.where(vel >= 0, back, fwd)

def ns_nonlin(w):
    u = biot_savart(w)
    out = torch.zeros_like(w)
    for c in range(3):
        adv = u[:, 0]*gup(u[:, 0], w[:, c], 'x')+u[:, 1]*gup(u[:, 1], w[:, c], 'y')+u[:, 2]*gup(u[:, 2], w[:, c], 'z')
        strc = w[:, 0]*gup(w[:, 0], u[:, c], 'x')+w[:, 1]*gup(w[:, 1], u[:, c], 'y')+w[:, 2]*gup(w[:, 2], u[:, c], 'z')
        out[:, c] = strc-adv
    return out

def l1w(alpha, Nt):
    j = torch.arange(Nt, device=DEV, dtype=DT)
    return (j+1.)**(1.-alpha)-torch.where(j == 0, torch.zeros_like(j), j**(1.-alpha))

def propagator(n, alpha, s, Nt):
    Z = symb(n, s).reshape(-1).double(); b = l1w(float(alpha), Nt).double()
    dt = T/Nt; dta = dt**float(alpha); g = torch.zeros((Nt+1, Z.numel()), device=DEV, dtype=torch.float64)
    g[0] = 1.; den = 1.+dta*NU*Z
    for k in range(1, Nt+1):
        base = b[k-1]*g[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV); base = base+torch.einsum('j,jy->y', b[k-j-1]-b[k-j], g[1:k])
        g[k] = base/den
    return g[Nt].reshape(n, n, n).to(DT)

def basef(w0, G): return ifftn(G[None]*fftn(w0))

def solve(w0, a0, s0, Nt, n, nonlin_on=True):
    B = w0.shape[0]; Z = symb(n, s0)[None]; b = l1w(a0, Nt)[None].repeat(B, 1)
    dt = T/Nt; dta = dt**float(a0); den = 1.+dta*NU*Z
    hist = torch.zeros((Nt+1, B, 3, n, n, n), device=DEV, dtype=DT); hist[0] = w0
    for k in range(1, Nt+1):
        base = b[:, k-1, None, None, None, None]*hist[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV); ww = b[:, k-j-1]-b[:, k-j]
            base = base+torch.einsum('bj,jbcxyz->bcxyz', ww, hist[1:k])
        nl = ns_nonlin(hist[k-1]) if nonlin_on else 0.
        hist[k] = ifftn(fftn(base+dta*nl)/den)
    return hist[Nt]

def smooth_vecpot(n, seed, kcut=5, std=1.0):
    rng = np.random.RandomState(seed); k = np.fft.fftfreq(n, d=1./n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij'); km = np.maximum(np.maximum(np.abs(kx), np.abs(ky)), np.abs(kz))
    amp = np.where(km <= kcut, (1.+km)**(-1.8), 0.)
    A = np.zeros((3, n, n, n), np.float32); wc = []
    for c in range(3):
        cc = (rng.randn(n, n, n)+1j*rng.randn(n, n, n))*amp; wc.append(cc)
    tc = lambda a: torch.tensor(a, device=DEV, dtype=torch.cfloat)
    tr = lambda a: torch.tensor(a, device=DEV, dtype=DT)
    Ah = [tc(wc[c]) for c in range(3)]
    kxt, kyt, kzt = tr(kx), tr(ky), tr(kz)
    wx = 1j*(kyt*Ah[2]-kzt*Ah[1]); wy = 1j*(kzt*Ah[0]-kxt*Ah[2]); wz = 1j*(kxt*Ah[1]-kyt*Ah[0])
    w = torch.stack([torch.fft.ifftn(x, dim=(-3,-2,-1), norm='ortho').real for x in (wx, wy, wz)])
    w = w/(w.std()+1e-12)*std
    return w.cpu().numpy().astype(np.float32)
def make_w0(n, seed): return smooth_vecpot(n, seed, std=2.5)

class SB3(nn.Module):
    def __init__(self, ci, co, kmod):
        super().__init__(); self.k = kmod
        self.W = nn.Parameter(torch.randn(co, ci, 2*kmod+1, 2*kmod+1, 2*kmod+1, dtype=torch.cfloat)*.02)
        self.loc = nn.Conv3d(ci, co, 1); self.b = nn.Parameter(torch.zeros(co))
    def forward(self, x):
        B, C, n, _, _ = x.shape; Xl = torch.fft.fftshift(fftn(x), dim=(-3,-2,-1))
        c = n//2; hw = min(self.k, n//2-1); sl = slice(c-hw, c+hw+1); ks = self.k
        Wc = self.W[:, :, ks-hw:ks+hw+1, ks-hw:ks+hw+1, ks-hw:ks+hw+1]
        Xs = torch.zeros(B, Wc.shape[0], n, n, n, dtype=torch.cfloat, device=x.device)
        Xs[:, :, sl, sl, sl] = torch.einsum('bixyz,oixyz->boxyz', Xl[:, :, sl, sl, sl], Wc)
        return torch.fft.ifftn(torch.fft.ifftshift(Xs, dim=(-3,-2,-1)), dim=(-3,-2,-1), norm='ortho').real+self.loc(x)+self.b[None, :, None, None, None]

class Net3(nn.Module):
    def __init__(self, cin=3, width=WIDTH, depth=DEPTH, kmod=KMOD, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.inp = nn.Conv3d(cin, width, 1); self.blocks = nn.ModuleList([SB3(width, width, kmod) for _ in range(depth)])
        self.out = nn.Conv3d(width, 3, 1)
        self.film = nn.ModuleList([nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, 2*width)) for _ in range(depth)])
        self.act = nn.GELU()
    def forward(self, x, af):
        v = self.inp(x); a = torch.tensor(af, device=x.device, dtype=x.dtype).reshape(1, 2).expand(x.shape[0], 2)
        for blk, fi in zip(self.blocks, self.film):
            v = self.act(blk(v)); gb = fi(a); g, bb = gb[:, :v.shape[1]], gb[:, v.shape[1]:]
            v = (1+g[:, :, None, None, None])*v+bb[:, :, None, None, None]
        return self.out(v)

def relL2(p, t): return float(torch.linalg.norm(p-t)/(torch.linalg.norm(t)+1e-12)*100.)

def build(n, Nt):
    W0 = np.stack([make_w0(n, 1000+i) for i in range(NTR)])
    SOL = np.zeros((NA, NS, NTR, 3, n, n, n), np.float32); LIN = np.zeros_like(SOL); Gt = {}
    w0t = torch.tensor(W0, device=DEV, dtype=DT)
    for ia, a in enumerate(A_GRID):
        for is_, s in enumerate(S_GRID):
            Gt[(ia, is_)] = propagator(n, a, s, Nt)
            with torch.no_grad():
                SOL[ia, is_] = solve(w0t, a, s, Nt, n, True).cpu().numpy()
                LIN[ia, is_] = solve(w0t, a, s, Nt, n, False).cpu().numpy()
    return W0, SOL, LIN, Gt

def train_one(kind, W0, SOL, G16, seed=1):
    torch.manual_seed(seed); np.random.seed(seed); net = Net3(seed=seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, 5e-5)
    w0t = torch.tensor(W0, device=DEV, dtype=DT); mean = float(SOL.mean()); sv = float(SOL.std()); rng = np.random.RandomState(123)
    t0 = time.time(); start = 0
    if os.path.exists(CKPT):
        d = torch.load(CKPT, map_location=DEV)
        if d.get('kind') == kind: net.load_state_dict(d['net']); opt.load_state_dict(d['opt']); start = d['step']
    for st in range(start, STEPS+1):
        ia, is_ = rng.randint(NA), rng.randint(NS); idx = rng.permutation(NTR)[:BATCH]
        w0 = w0t[idx]; sol = torch.tensor(SOL[ia, is_, idx], device=DEV, dtype=DT)
        with torch.no_grad(): bf = basef(w0, G16[(ia, is_)])
        out = net(w0, (A_GRID[ia], S_GRID[is_])); target = (sol-bf)/sv if kind == 'A' else (sol-mean)/sv
        loss = F.mse_loss(out, target); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if st % 500 == 0: torch.save({'kind': kind, 'step': st, 'net': net.state_dict(), 'opt': opt.state_dict()}, CKPT)
    if os.path.exists(CKPT): os.remove(CKPT)
    net.eval(); return net, time.time()-t0, mean, sv

_GC = {}
@torch.no_grad()
def evaluate(net, kind, n, Nt, We, a_te, s_te, mean, sv):
    ev = []
    for (a, s) in sorted(set(zip(np.round(a_te, 4).tolist(), np.round(s_te, 4).tolist()))):
        idx = [b for b in range(len(We)) if abs(a_te[b]-a) < 1e-6 and abs(s_te[b]-s) < 1e-6]
        w0 = torch.tensor(We[idx], device=DEV, dtype=DT); sol = solve(w0, a, s, Nt, n, True)
        out = net(w0, (a, s)); key = (n, round(a, 4), round(s, 4), Nt)
        if key not in _GC: _GC[key] = propagator(n, a, s, Nt)
        pred = basef(w0, _GC[key])+sv*out if kind == 'A' else mean+sv*out
        for j in range(len(idx)): ev.append(relL2(pred[j], sol[j]))
    return float(np.mean(ev))

def main():
    tg = time.time()
    log('3D incompressible fractional NS, vector vorticity (with vortex stretching), T=%g nu=%g' % (T, NU))
    with torch.no_grad():
        w = torch.tensor(np.stack([make_w0(12, 1), make_w0(12, 2)]), device=DEV, dtype=DT)
        kx, ky, kz, _ = waves(12); divw = ifftn(1j*(kx[None]*fftn(w)[:, 0]+ky[None]*fftn(w)[:, 1]+kz[None]*fftn(w)[:, 2]))
        u = biot_savart(w); divu = ifftn(1j*(kx[None]*fftn(u)[:, 0]+ky[None]*fftn(u)[:, 1]+kz[None]*fftn(u)[:, 2]))
        log('div(w)=%.2e (curl of vector potential), Biot-Savart div(u)=%.2e' % (float(divw.abs().max()), float(divu.abs().max())))
        G = propagator(12, .85, .85, 120); lin = solve(w, .85, .85, 120, 12, False)
        log('linear propagator consistency %.2e' % float(torch.linalg.norm(lin-basef(w, G))/torch.linalg.norm(lin)))
    n0, nt0 = RES[0]; W0, SOL, LIN, G16 = build(n0, nt0)
    nl = float(np.linalg.norm(SOL-LIN)/(np.linalg.norm(LIN)+1e-12)*100)
    log('regime: stretching+advection change the linear solution by %.1f%%; params=%gK' % (nl, sum(p.numel() for p in Net3().parameters())/1e3)); save_log()
    rng = np.random.RandomState(7); seeds = rng.randint(100000, size=NTE)
    a_te = A_GRID[rng.randint(NA, size=NTE)]; s_te = S_GRID[rng.randint(NS, size=NTE)]
    res = {}
    for kind, tag, sd in [('A', 'FrFNO', 1), ('F', 'FNO', 2)]:
        net, tt, mean, sv = train_one(kind, W0, SOL, G16, seed=sd); row = {'train_s': tt}
        for n, Nt in RES:
            We = np.stack([make_w0(n, int(x)) for x in seeds]); row['n%d' % n] = evaluate(net, kind, n, Nt, We, a_te, s_te, mean, sv)
        res[tag] = row; log('[%s] vector-vorticity relL2 %.3f/%.3f/%.3f (12/16/24) %.1fs' % (tag, row['n12'], row['n16'], row['n24'], tt)); save_log()
    log('\n===== 3D vector-vorticity fractional NS: relL2 (%) =====')
    for tag in res:
        r = res[tag]; log('%-6s %8.3f %8.3f %8.3f  train %.1fs' % (tag, r['n12'], r['n16'], r['n24'], r['train_s']))
    np.savez(os.path.join(CKDIR, 'ns3d_metrics.npz'), results=json.dumps(res)); log('total %.1fs' % (time.time()-tg)); save_log()

if __name__ == '__main__': main()
