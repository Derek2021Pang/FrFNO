# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
Step 1 of the systems extension: 2D TWO-COMPONENT space-time fractional
vector Burgers system on Omega=(0,1)^2 with homogeneous Dirichlet BC.

  ^C D_t^a u = -nu_u(x)(-D)^s u - (u d_x u + v d_y u)
  ^C D_t^a v = -nu_v(x)(-D)^s v - (u d_x v + v d_y v)

The two components have DIFFERENT variable diffusivities (diagonal, case-2
propagator) and are coupled through the vector advection (U.grad)U. The frozen
linear propagator is the per-component Mittag-Leffler multiplier (same c0=1 for
both); variable diffusivity and the whole vector nonlinearity form the residual.

One-knife: FrFNO (exact propagator + learned 2-channel residual) vs matched FNO
(full 2-channel field); same backbone (~same param count), optimizer, 3000 steps.
Train internal 15^2, zero-shot 31^2/63^2. Dirichlet DST-I + five-point symbol,
L1 implicit diffusion skeleton (Picard n_inner=2), explicit upwind advection.
"""
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, _PKG_ROOT)
from gpu_solver import dst2_torch, spectral_eigvals_torch, l1_b_torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float32
M = 2
torch.manual_seed(0); np.random.seed(0)

T = 0.05
A_GRID = np.array([.7, .85, 1.], np.float32)
S_GRID = np.array([.5, .75, 1.], np.float32)
NA, NS = len(A_GRID), len(S_GRID)
NTR, NTE, STEPS, BATCH = 64, 24, 3000, 64
WIDTH, DEPTH, KMOD = 32, 4, 7
RES = [(16, 400), (32, 800), (64, 1600)]
C0 = 0.25
ROOT = _PKG_ROOT
CKDIR = ROOT
OUT = os.path.join(CKDIR, 'system_burgers_result.txt')
LOG = []
def log(m): print(m, flush=True); LOG.append(str(m))
def save_log(): open(OUT, 'w', encoding='utf-8').write('\n'.join(LOG))

def Qs_field(Nx, s): return spectral_eigvals_torch(Nx, DEV, DT) ** float(s)
def l1_b(alpha, Nt): return l1_b_torch(float(alpha), Nt, DEV, DT)

def lin_propagator(Nx, alpha, s, Nt, c=C0):
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
            base = base + torch.einsum('j,jy->y', b[k-j-1]-b[k-j], g[1:k])
        g[k] = base/denom
    return g[Nt].reshape(n, n).to(DT)

def base_field(U0, G):
    # U0: (B,M,n,n); apply DST multiplier per component (flatten component into batch)
    B, Mm = U0.shape[0], U0.shape[1]; n = U0.shape[-1]
    flat = U0.reshape(B*Mm, n, n)
    out = dst2_torch(G*dst2_torch(flat))
    return out.reshape(B, Mm, n, n)

def upwind(U, q, h):
    # -(U_x d_x q + U_y d_y q); U,q: (B,2,n,n); first channel=u(x-dir), second=v(y-dir)
    u, v = U[:, 0], U[:, 1]
    qp = F.pad(q, (1, 1, 1, 1))                      # zero outside = Dirichlet
    qc = qp[:, 1:-1, 1:-1]
    dxm = (qc-qp[:, 1:-1, :-2])/h; dxp = (qp[:, 1:-1, 2:]-qc)/h
    dym = (qc-qp[:, :-2, 1:-1])/h; dyp = (qp[:, 2:, 1:-1]-qc)/h
    dxq = torch.where(u >= 0, dxm, dxp); dyq = torch.where(v >= 0, dym, dyp)
    return -(u*dxq + v*dyq)

def sys_solve(U0, NU, a0, s0, Nx, Nt, adv_on=True, n_inner=2):
    # U0,NU: (B,M,n,n). Flatten component into batch for the DST diffusion solve.
    B = U0.shape[0]; n = Nx-1; BM = B*M
    uf = U0.reshape(BM, n, n); nuf = NU.reshape(BM, n, n)
    Qs = Qs_field(Nx, s0)[None]
    b = l1_b(a0, Nt)[None].repeat(BM, 1)
    dt = T/Nt; dta = dt**float(a0); h = 1.0/Nx
    c = nuf.amax(dim=(1, 2), keepdim=True)
    denom = b[:, 0, None, None]+dta*c*Qs
    hist = torch.zeros((Nt+1, BM, n, n), device=DEV, dtype=DT)
    hist[0] = uf
    for k in range(1, Nt+1):
        base = b[:, k-1, None, None]*hist[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV)
            w = b[:, k-j-1]-b[:, k-j]
            base = base + torch.einsum('bj,jbxy->bxy', w, hist[1:k])
        x = hist[k-1]
        adv = torch.zeros_like(x)
        if adv_on:
            xm = x.reshape(B, M, n, n)
            au = upwind(xm, xm[:, 0], h); av = upwind(xm, xm[:, 1], h)
            adv = torch.stack([au, av], 1).reshape(BM, n, n)
        for _ in range(n_inner):
            Lx = dst2_torch(Qs*dst2_torch(x))
            r = base - dta*(nuf-c)*Lx + dta*adv
            x = dst2_torch(dst2_torch(r)/denom)
        hist[k] = x
    return hist[Nt].reshape(B, M, n, n)

def sine_field(Nx, seed, kcut=6, decay=1.8, std=0.18, mean=0.60):
    n = Nx-1; xi = np.arange(1, Nx)/Nx
    X, Y = np.meshgrid(xi, xi, indexing='ij'); rng = np.random.RandomState(seed)
    v = np.zeros((n, n), np.float64)
    for kx in range(1, kcut+1):
        for ky in range(1, kcut+1):
            amp = (kx**2+ky**2)**(-decay/2.0)
            v += amp*(rng.randn()*np.sin(kx*np.pi*X)*np.sin(ky*np.pi*Y))
    v = (v-v.mean())/(v.std()+1e-12)*std+mean
    return v.astype(np.float32)

def nu_field(Nx, seed, lo=.6, hi=1.4):
    g = sine_field(Nx, seed, kcut=4, decay=1.5, std=1.0, mean=0.0)
    return (0.5*(lo+hi)+0.5*(hi-lo)*g/(np.abs(g).max()+1e-12)).astype(np.float32)

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

class SysNet(nn.Module):
    def __init__(self, cin=2*M, width=WIDTH, depth=DEPTH, kmod=KMOD, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.inp = nn.Conv2d(cin, width, 1)
        self.blocks = nn.ModuleList([SpecBlock(width, width, kmod) for _ in range(depth)])
        self.out = nn.Conv2d(width, M, 1)
        self.film = nn.ModuleList([nn.Sequential(nn.Linear(2, width), nn.GELU(),
                              nn.Linear(width, 2*width)) for _ in range(depth)])
        self.act = nn.GELU()
    def forward(self, x, af):
        v = self.inp(x)
        a = torch.tensor(af, device=x.device, dtype=x.dtype).reshape(1, 2).expand(x.shape[0], 2)
        for blk, fi in zip(self.blocks, self.film):
            v = self.act(blk(v)); gb = fi(a); g, bb = gb[:, :v.shape[1]], gb[:, v.shape[1]:]
            v = (1+g[:, :, None, None])*v+bb[:, :, None, None]
        return self.out(v)                                   # (B,M,H,W)

def relL2(p, t): return float(torch.linalg.norm(p-t)/(torch.linalg.norm(t)+1e-12)*100.0)

def build_train(Nx, Nt):
    n = Nx-1
    U0 = np.stack([np.stack([sine_field(Nx, 1000+2*i), sine_field(Nx, 1000+2*i+1)]) for i in range(NTR)])
    NU = np.stack([np.stack([nu_field(Nx, 5000+2*i, .15, .35), nu_field(Nx, 5000+2*i+1, .20, .40)]) for i in range(NTR)])
    SOL = np.zeros((NA, NS, NTR, M, n, n), np.float32); LIN = np.zeros_like(SOL); G = {}
    for ia, a in enumerate(A_GRID):
        for is_, s in enumerate(S_GRID):
            G[(ia, is_)] = lin_propagator(Nx, a, s, Nt)
            with torch.no_grad():
                U0t = torch.tensor(U0, device=DEV, dtype=DT); NUt = torch.tensor(NU, device=DEV, dtype=DT)
                SOL[ia, is_] = sys_solve(U0t, NUt, a, s, Nx, Nt, True).cpu().numpy()
                LIN[ia, is_] = sys_solve(U0t, NUt, a, s, Nx, Nt, False).cpu().numpy()
    return U0, NU, SOL, LIN, G

def train_one(mk, U0, NU, SOL, G, Nx, Nt, seed=1):
    torch.manual_seed(seed); np.random.seed(seed)
    net = SysNet(seed=seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, 5e-5)
    U0t = torch.tensor(U0, device=DEV, dtype=DT); NUt = torch.tensor(NU, device=DEV, dtype=DT)
    mean = float(SOL.mean()); sv = float(SOL.std()); rng = np.random.RandomState(123)
    ck = os.path.join(CKDIR, 'sys_ckpt_%s.pt' % mk); t0 = time.time(); start = 0
    if os.path.exists(ck):
        d = torch.load(ck, map_location=DEV); net.load_state_dict(d['net']); opt.load_state_dict(d['opt']); start = d['step']
    for st in range(start, STEPS+1):
        ia, is_ = rng.randint(NA), rng.randint(NS); idx = rng.permutation(NTR)[:BATCH]
        u0 = U0t[idx]; nu = NUt[idx]; sol = torch.tensor(SOL[ia, is_, idx], device=DEV, dtype=DT)
        with torch.no_grad(): base = base_field(u0, G[(ia, is_)])
        xb = torch.cat([u0, nu], 1); out = net(xb, (A_GRID[ia], S_GRID[is_]))
        target = (sol-base)/sv if mk == 'A' else (sol-mean)/sv
        loss = F.mse_loss(out, target)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if st % 500 == 0: torch.save({'step': st, 'net': net.state_dict(), 'opt': opt.state_dict()}, ck)
    if os.path.exists(ck): os.remove(ck)
    net.eval(); return net, time.time()-t0, mean, sv

_GCACHE = {}
@torch.no_grad()
def evaluate(net, mk, Nx, Nt, Ue, Ne, a_te, s_te, mean, sv):
    eu, ev = [], []
    keys = sorted(set(zip(np.round(a_te, 4).tolist(), np.round(s_te, 4).tolist())))
    for (a, s) in keys:
        idx = [b for b in range(len(Ue)) if abs(a_te[b]-a) < 1e-6 and abs(s_te[b]-s) < 1e-6]
        u0 = torch.tensor(Ue[idx], device=DEV, dtype=DT); nu = torch.tensor(Ne[idx], device=DEV, dtype=DT)
        sol = sys_solve(u0, nu, a, s, Nx, Nt, True)
        out = net(torch.cat([u0, nu], 1), (a, s))
        if mk == 'A':
            key = (Nx, round(a, 4), round(s, 4), Nt)
            if key not in _GCACHE: _GCACHE[key] = lin_propagator(Nx, a, s, Nt)
            pred = base_field(u0, _GCACHE[key])+sv*out
        else: pred = mean+sv*out
        for j in range(len(idx)):
            eu.append(relL2(pred[j, 0], sol[j, 0])); ev.append(relL2(pred[j, 1], sol[j, 1]))
    return float(np.mean(eu)), float(np.mean(ev)), float(np.mean(eu+ev))

def main():
    tg = time.time()
    log('2D two-component fractional vector Burgers system (Dirichlet DST, T=%g)' % T)
    log('train 15^2 NTR=%d steps=%d order %dx%d NTE=%d res=%s' % (NTR, STEPS, NA, NS, NTE, RES))
    Nx0, Nt0 = RES[0]
    U0, NU, SOL, LIN, G16 = build_train(Nx0, Nt0)
    chg = float(np.linalg.norm(SOL[1, 1]-LIN[1, 1])/(np.linalg.norm(LIN[1, 1])+1e-12)*100)
    log('regime: vector advection (U.grad)U changes the linear solution by %.1f%% (a=.85,s=.75)' % chg)
    save_log()
    rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=NTE)
    ia_te = rng.randint(NA, size=NTE); is_te = rng.randint(NS, size=NTE)
    a_te = A_GRID[ia_te]; s_te = S_GRID[is_te]
    res = {}
    for mk, tag, sd in [('A', 'FrFNO', 1), ('F', 'FNO', 2)]:
        net, tt, mean, sv = train_one(mk, U0, NU, SOL, G16, Nx0, Nt0, seed=sd)
        row = {'train_s': tt}
        for Nx, Nt in RES:
            Ue = np.stack([np.stack([sine_field(Nx, int(s2)), sine_field(Nx, int(s2)+1)]) for s2 in seeds])
            Ne = np.stack([np.stack([nu_field(Nx, int(s2), .15, .35), nu_field(Nx, int(s2)+1, .20, .40)]) for s2 in seeds])
            eu, ev, em = evaluate(net, mk, Nx, Nt, Ue, Ne, a_te, s_te, mean, sv)
            row['nx%d' % Nx] = (eu, ev, em)
        res[tag] = row
        log('[%s] u: %.3f/%.3f/%.3f  v: %.3f/%.3f/%.3f  mean: %.3f/%.3f/%.3f (%.1fs)' % (
            tag, row['nx16'][0], row['nx32'][0], row['nx64'][0],
            row['nx16'][1], row['nx32'][1], row['nx64'][1],
            row['nx16'][2], row['nx32'][2], row['nx64'][2], tt)); save_log()
    log('\n===== two-component system summary: component-averaged relL2 (%) =====')
    log('%-6s %9s %9s %9s %10s' % ('model', '16', '32(ZS)', '64(ZS)', 'train_s'))
    for tag in ['FrFNO', 'FNO']:
        r = res[tag]
        log('%-6s %8.3f%% %8.3f%% %8.3f%% %9.1f' % (tag, r['nx16'][2], r['nx32'][2], r['nx64'][2], r['train_s']))
    np.savez(os.path.join(CKDIR, 'system_burgers_metrics.npz'), results=json.dumps(res))
    log('total %.1fs' % (time.time()-tg)); save_log()

if __name__ == '__main__': main()
