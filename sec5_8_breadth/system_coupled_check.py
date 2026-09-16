# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
Step (iii) of the systems extension: two-component fractional system with a
NON-SYMMETRIC component-coupling matrix J.

  ^C D_t^a U = -nu(x)(-D)^s U - J U + N(U),  U=(u,v),
  N(U) = -(U.grad)U (weak vector advection, upwind), nu shared by components.

The frozen linear principal part at mode lambda_k is the 2x2 matrix
L_k = c0 lambda_k^s I + J. Because J is constant it is diagonalized ONCE,
J = V diag(mu1,mu2) V^{-1}; in the eigen-coordinates W = V^{-1} U the linear
part decouples into two scalar problems with symbols (c0 lambda^s + mu_i), each
solved by the scalar Mittag-Leffler/L1 multiplier, and the base is rotated back
U_base = V diag(G1,G2) V^{-1} U0 -- the discrete form of the matrix ML
E_a(-t^a L_k)=V diag(E_a) V^{-1}.

Three models, identical 2-channel backbone/budget:
  A (FrFNO-full): exact matrix (eigen-rotated) propagator;
  D (ablation)  : propagator that WRONGLY drops the off-diagonal coupling
                  (uses diag(J), no rotation) -- tests whether the full matrix
                  propagator is actually necessary;
  F (FNO)       : no propagator, learns the full field.
Dirichlet DST-I, five-point symbol, L1 implicit linear part, explicit upwind.
"""
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, _PKG_ROOT)
from gpu_solver import dst2_torch, spectral_eigvals_torch, l1_b_torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float32; M = 2
torch.manual_seed(0); np.random.seed(0)

T = 0.05; C0 = 1.0
# non-symmetric J with two positive REAL eigenvalues (uniformly dissipative)
JNP = np.array([[1.2, 0.8], [0.3, 0.6]], np.float64)
_mu, _V = np.linalg.eig(JNP)
assert np.all(np.abs(_mu.imag) < 1e-8) and np.all(_mu.real > 0), (_mu, JNP)
MU = [float(_mu[0].real), float(_mu[1].real)]
V = torch.tensor(_V.real, device=DEV, dtype=DT)
VINV = torch.linalg.inv(V)
JD = np.diag(JNP).astype(np.float32)          # wrong (diagonal-only) ablation
A_GRID = np.array([.7, .85, 1.], np.float32); S_GRID = np.array([.5, .75, 1.], np.float32)
NA, NS = len(A_GRID), len(S_GRID)
NTR, NTE, STEPS, BATCH = 64, 24, 3000, 64
WIDTH, DEPTH, KMOD = 32, 4, 7
RES = [(16, 400), (32, 800), (64, 1600)]
ROOT = _PKG_ROOT
CKDIR = ROOT; OUT = os.path.join(CKDIR, 'system_coupled_result.txt')
LOG = []
def log(m): print(m, flush=True); LOG.append(str(m))
def save_log(): open(OUT, 'w', encoding='utf-8').write('\n'.join(LOG))

def Qs_field(Nx, s): return spectral_eigvals_torch(Nx, DEV, DT)**float(s)
def l1_b(alpha, Nt): return l1_b_torch(float(alpha), Nt, DEV, DT)

def scalar_G(Nx, alpha, s, Nt, mu, c=C0):
    """frozen L1 terminal multiplier for symbol c*lambda^s + mu."""
    n = Nx-1
    Z = (c*Qs_field(Nx, s) + mu).reshape(-1).double()
    b = l1_b_torch(float(alpha), Nt, DEV, torch.float64)
    dt = T/Nt; dta = dt**float(alpha)
    g = torch.zeros((Nt+1, Z.numel()), device=DEV, dtype=torch.float64)
    g[0] = 1.0; denom = b[0]+dta*Z
    for k in range(1, Nt+1):
        base = b[k-1]*g[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV, dtype=torch.long)
            base = base + torch.einsum('j,jy->y', b[k-j-1]-b[k-j], g[1:k])
        g[k] = base/denom
    return g[Nt].reshape(n, n).to(DT)

def rotate(P, U):  # P (2,2) times component axis: w_i = sum_j P[i,j] U_j
    return torch.einsum('ij,bjxy->bixy', P, U)

def base_full(U0, Gs):
    W = rotate(VINV, U0)
    wb = torch.stack([dst2_torch(Gs[i]*dst2_torch(W[:, i])) for i in range(M)], 1)
    return rotate(V, wb)

def base_diag(U0, Gd):  # wrong: ignore off-diagonal, per-component with J_ii
    return torch.stack([dst2_torch(Gd[i]*dst2_torch(U0[:, i])) for i in range(M)], 1)

def upwind(U, q, h):
    u, v = U[:, 0], U[:, 1]; qp = F.pad(q, (1, 1, 1, 1)); qc = qp[:, 1:-1, 1:-1]
    dxm = (qc-qp[:, 1:-1, :-2])/h; dxp = (qp[:, 1:-1, 2:]-qc)/h
    dym = (qc-qp[:, :-2, 1:-1])/h; dyp = (qp[:, 2:, 1:-1]-qc)/h
    return -(u*torch.where(u >= 0, dxm, dxp) + v*torch.where(v >= 0, dym, dyp))

def sys_solve(U0, nu, a0, s0, Nx, Nt, adv_on=True, n_inner=2):
    # U0 (B,2,n,n); nu (B,n,n) shared; integrate in J-eigenbasis W=V^-1 U
    B = U0.shape[0]; n = Nx-1; BM = B*M
    W0 = rotate(VINV, U0).reshape(BM, n, n)
    nuf = nu.unsqueeze(1).repeat(1, M, 1, 1).reshape(BM, n, n)
    Qs = Qs_field(Nx, s0)[None]; b = l1_b(a0, Nt)[None].repeat(BM, 1)
    dt = T/Nt; dta = dt**float(a0); h = 1.0/Nx
    c = nuf.amax(dim=(1, 2), keepdim=True)
    muf = torch.tensor(MU, device=DEV, dtype=DT).reshape(M, 1, 1, 1).repeat(B, 1, 1, 1).reshape(BM, 1, 1)
    denom = b[:, 0, None, None]+dta*(c*Qs+muf)
    hist = torch.zeros((Nt+1, BM, n, n), device=DEV, dtype=DT); hist[0] = W0
    for k in range(1, Nt+1):
        base = b[:, k-1, None, None]*hist[0]
        if k > 1:
            j = torch.arange(1, k, device=DEV); w = b[:, k-j-1]-b[:, k-j]
            base = base + torch.einsum('bj,jbxy->bxy', w, hist[1:k])
        x = hist[k-1]; advW = torch.zeros_like(x)
        if adv_on:
            Up = rotate(V, x.reshape(B, M, n, n))
            adv = torch.stack([upwind(Up, Up[:, 0], h), upwind(Up, Up[:, 1], h)], 1)
            advW = rotate(VINV, adv).reshape(BM, n, n)
        for _ in range(n_inner):
            Lx = dst2_torch(Qs*dst2_torch(x))
            r = base - dta*(nuf-c)*Lx + dta*advW
            x = dst2_torch(dst2_torch(r)/denom)
        hist[k] = x
    return rotate(V, hist[Nt].reshape(B, M, n, n))

def sine_field(Nx, seed, kcut=6, decay=1.8, std=0.18, mean=0.60):
    n = Nx-1; xi = np.arange(1, Nx)/Nx
    X, Y = np.meshgrid(xi, xi, indexing='ij'); rng = np.random.RandomState(seed); v = np.zeros((n, n), np.float64)
    for kx in range(1, kcut+1):
        for ky in range(1, kcut+1):
            v += (kx**2+ky**2)**(-decay/2.0)*(rng.randn()*np.sin(kx*np.pi*X)*np.sin(ky*np.pi*Y))
    return ((v-v.mean())/(v.std()+1e-12)*std+mean).astype(np.float32)

def nu_field(Nx, seed, lo=.8, hi=1.2):
    g = sine_field(Nx, seed, 4, 1.5, 1.0, 0.0)
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
        return torch.fft.ifft2(torch.fft.ifftshift(Xs, dim=(-2, -1)), norm='ortho').real+self.loc(x)+self.b[None, :, None, None]

class SysNet(nn.Module):
    def __init__(self, cin=3, width=WIDTH, depth=DEPTH, kmod=KMOD, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.inp = nn.Conv2d(cin, width, 1)
        self.blocks = nn.ModuleList([SpecBlock(width, width, kmod) for _ in range(depth)])
        self.out = nn.Conv2d(width, M, 1)
        self.film = nn.ModuleList([nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, 2*width)) for _ in range(depth)])
        self.act = nn.GELU()
    def forward(self, x, af):
        v = self.inp(x); a = torch.tensor(af, device=x.device, dtype=x.dtype).reshape(1, 2).expand(x.shape[0], 2)
        for blk, fi in zip(self.blocks, self.film):
            v = self.act(blk(v)); gb = fi(a); g, bb = gb[:, :v.shape[1]], gb[:, v.shape[1]:]
            v = (1+g[:, :, None, None])*v+bb[:, :, None, None]
        return self.out(v)

def relL2(p, t): return float(torch.linalg.norm(p-t)/(torch.linalg.norm(t)+1e-12)*100.0)

def build_train(Nx, Nt):
    n = Nx-1
    U0 = np.stack([np.stack([sine_field(Nx, 1000+2*i), sine_field(Nx, 1000+2*i+1)]) for i in range(NTR)])
    NU = np.stack([nu_field(Nx, 5000+i) for i in range(NTR)])
    SOL = np.zeros((NA, NS, NTR, M, n, n), np.float32); NOJ = np.zeros_like(SOL); GF, GD = {}, {}
    for ia, a in enumerate(A_GRID):
        for is_, s in enumerate(S_GRID):
            GF[(ia, is_)] = [scalar_G(Nx, a, s, Nt, MU[i]) for i in range(M)]
            GD[(ia, is_)] = [scalar_G(Nx, a, s, Nt, float(JD[i])) for i in range(M)]
            with torch.no_grad():
                Ut = torch.tensor(U0, device=DEV, dtype=DT); nt = torch.tensor(NU, device=DEV, dtype=DT)
                SOL[ia, is_] = sys_solve(Ut, nt, a, s, Nx, Nt, True).cpu().numpy()
                # coupling effect: same but J=0 linear part (mu=0) for regime reporting
                global MU
                old = MU[:]; MU[:] = [0., 0.]
                NOJ[ia, is_] = sys_solve(Ut, nt, a, s, Nx, Nt, True).cpu().numpy(); MU[:] = old
    return U0, NU, SOL, NOJ, GF, GD

def train_one(mk, U0, NU, SOL, GF, GD, Nx, Nt, seed=1):
    torch.manual_seed(seed); np.random.seed(seed); net = SysNet(seed=seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, 5e-5)
    U0t = torch.tensor(U0, device=DEV, dtype=DT); NUt = torch.tensor(NU, device=DEV, dtype=DT)
    mean = float(SOL.mean()); sv = float(SOL.std()); rng = np.random.RandomState(123)
    ck = os.path.join(CKDIR, 'sysj_ckpt_%s.pt' % mk); t0 = time.time(); start = 0
    if os.path.exists(ck):
        d = torch.load(ck, map_location=DEV); net.load_state_dict(d['net']); opt.load_state_dict(d['opt']); start = d['step']
    for st in range(start, STEPS+1):
        ia, is_ = rng.randint(NA), rng.randint(NS); idx = rng.permutation(NTR)[:BATCH]
        u0 = U0t[idx]; nu = NUt[idx]; sol = torch.tensor(SOL[ia, is_, idx], device=DEV, dtype=DT)
        with torch.no_grad():
            bf = base_full(u0, GF[(ia, is_)]); bd = base_diag(u0, GD[(ia, is_)])
        xb = torch.cat([u0, nu.unsqueeze(1)], 1); out = net(xb, (A_GRID[ia], S_GRID[is_]))
        if mk == 'A': target = (sol-bf)/sv
        elif mk == 'D': target = (sol-bd)/sv
        else: target = (sol-mean)/sv
        loss = F.mse_loss(out, target); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if st % 500 == 0: torch.save({'step': st, 'net': net.state_dict(), 'opt': opt.state_dict()}, ck)
    if os.path.exists(ck): os.remove(ck)
    net.eval(); return net, time.time()-t0, mean, sv

_GC = {}
@torch.no_grad()
def evaluate(net, mk, Nx, Nt, Ue, Ne, a_te, s_te, mean, sv):
    eu, ev = [], []
    for (a, s) in sorted(set(zip(np.round(a_te, 4).tolist(), np.round(s_te, 4).tolist()))):
        idx = [b for b in range(len(Ue)) if abs(a_te[b]-a) < 1e-6 and abs(s_te[b]-s) < 1e-6]
        u0 = torch.tensor(Ue[idx], device=DEV, dtype=DT); nu = torch.tensor(Ne[idx], device=DEV, dtype=DT)
        sol = sys_solve(u0, nu, a, s, Nx, Nt, True); out = net(torch.cat([u0, nu.unsqueeze(1)], 1), (a, s))
        key = (Nx, round(a, 4), round(s, 4), Nt)
        if key not in _GC:
            _GC[key] = ([scalar_G(Nx, a, s, Nt, MU[i]) for i in range(M)],
                        [scalar_G(Nx, a, s, Nt, float(JD[i])) for i in range(M)])
        Gf, Gd = _GC[key]
        if mk == 'A': pred = base_full(u0, Gf)+sv*out
        elif mk == 'D': pred = base_diag(u0, Gd)+sv*out
        else: pred = mean+sv*out
        for j in range(len(idx)): eu.append(relL2(pred[j, 0], sol[j, 0])); ev.append(relL2(pred[j, 1], sol[j, 1]))
    return float(np.mean(eu)), float(np.mean(ev)), float(np.mean(eu+ev))

def main():
    tg = time.time()
    log('Case (iii): 2-comp fractional system with non-symmetric J=%s' % JNP.tolist())
    log('eigenvalues mu=%s (positive real); V real; T=%g c0=%g' % ([round(x, 4) for x in MU], T, C0))
    Nx0, Nt0 = RES[0]; U0, NU, SOL, NOJ, GF, GD = build_train(Nx0, Nt0)
    coup = float(np.linalg.norm(SOL[1, 1]-NOJ[1, 1])/(np.linalg.norm(NOJ[1, 1])+1e-12)*100)
    log('regime: non-symmetric coupling J changes the J=0 linear solution by %.1f%%' % coup)
    save_log()
    rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=NTE)
    a_te = A_GRID[rng.randint(NA, size=NTE)]; s_te = S_GRID[rng.randint(NS, size=NTE)]
    res = {}
    for mk, tag, sd in [('A', 'FrFNO-full', 1), ('D', 'FrFNO-diag(ablation)', 3), ('F', 'FNO', 2)]:
        net, tt, mean, sv = train_one(mk, U0, NU, SOL, GF, GD, Nx0, Nt0, seed=sd); row = {'train_s': tt}
        for Nx, Nt in RES:
            Ue = np.stack([np.stack([sine_field(Nx, int(s2)), sine_field(Nx, int(s2)+1)]) for s2 in seeds])
            Ne = np.stack([nu_field(Nx, int(s2)) for s2 in seeds])
            row['nx%d' % Nx] = evaluate(net, mk, Nx, Nt, Ue, Ne, a_te, s_te, mean, sv)
        res[tag] = row
        r16, r32, r64 = row['nx16'], row['nx32'], row['nx64']
        log('[%s] mean %.3f/%.3f/%.3f (u %.3f/%.3f/%.3f v %.3f/%.3f/%.3f) %.1fs' % (
            tag, r16[2], r32[2], r64[2], r16[0], r32[0], r64[0], r16[1], r32[1], r64[1], tt)); save_log()
    log('\n===== non-symmetric coupled system: component-mean relL2 (%) =====')
    log('%-22s %9s %9s %9s %9s' % ('model', '16', '32(ZS)', '64(ZS)', 'train_s'))
    for tag in res:
        r = res[tag]; log('%-22s %8.3f%% %8.3f%% %8.3f%% %8.1f' % (tag, r['nx16'][2], r['nx32'][2], r['nx64'][2], r['train_s']))
    np.savez(os.path.join(CKDIR, 'system_coupled_metrics.npz'), results=json.dumps(res)); log('total %.1fs' % (time.time()-tg)); save_log()

if __name__ == '__main__': main()
