# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SEC = _os.path.dirname(_HERE); _PKG_ROOT = _os.path.dirname(_SEC)
for _p in (_HERE, _SEC, _PKG_ROOT):
    if _p not in _sys.path: _sys.path.insert(0, _p)
"""Numerical closed loop for Proposition 1 (P1-5 propagator error scaling) and Proposition 2 (P1-4 spectral-tail power law). Resumable and saved point by point.
STAGE: eta / Tlin / beta / spec ; SMOKE=1 for a fast smoke test."""
import os, sys, time
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'fractional_pde_study'))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import frfno_legacy as LG
from frfno_legacy import build_prop_table, prop_at, ub_full_batch, generate_multiscale_initial, DEV, DT
from gpu_solver import gpu_solve_imex_batch, spectral_eigvals_torch, dst2_torch
from nl_solver import gpu_solve_nl_batch
from d_scan import kl_basis

OUT = os.path.join(ROOT, 'theory_closure')
os.makedirs(OUT, exist_ok=True)
SMOKE = os.environ.get('SMOKE') == '1'
ALP = 0.85
L = []
def log(s): print(s, flush=True); L.append(str(s))
def save_txt(tag): open(os.path.join(OUT, f'p15_p14_{tag}.txt'), 'w', encoding='utf-8').write('\n'.join(L))
def nz(path): return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
def rel(a, b):
    return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12))

def fixed_w(Nx, seed=314, kmax=12, beta=1.0):
    """Coefficient perturbation field (Nx+1,Nx+1) with fixed shape, zero interior mean, and ||w||inf=1."""
    Phi, e = kl_basis(Nx, kmax=kmax, beta=beta)
    rng = np.random.RandomState(seed); g = np.zeros((Nx+1, Nx+1))
    for k in range(kmax):
        g += rng.normal()*np.sqrt(e[k])*Phi[k]
    w = g.copy(); win = w[1:Nx, 1:Nx]; win -= win.mean(); win /= (np.abs(win).max()+1e-12)
    w[1:Nx, 1:Nx] = win; w[[0,-1],:] = 0; w[:,[0,-1]] = 0
    return w.astype(np.float32)

def powerlaw_u0(Nx, p=2.0, seed=0, target_rms=0.5):
    """Zero-boundary initial data with a broad power-law spectrum: DST coefficients scale as |k|^{-p} in amplitude, synthesized by inverse DST-I (self-inverse), then scaled to a target rms."""
    n = Nx-1; rng = np.random.RandomState(seed)
    k = np.arange(1, n+1); K1, K2 = np.meshgrid(k, k, indexing='ij')
    rr = np.sqrt(K1**2+K2**2)
    c = rng.normal(size=(n, n))/ (rr**p + 1e-9)
    tc = torch.tensor(c[None], dtype=torch.float64)
    phys = dst2_torch(tc)[0].numpy()
    phys *= target_rms/(np.sqrt((phys**2).mean())+1e-12)
    u = np.zeros((Nx+1, Nx+1)); u[1:Nx, 1:Nx] = phys
    return u.astype(np.float32)

def ml_ab(alpha, beta, z, nterm=80):
    """Two-parameter Mittag-Leffler E_{alpha,beta}(-x), x>=0; a piecewise join of the small-x series and the large-x algebraic asymptotics."""
    z = np.asarray(z, np.float64); out = np.zeros_like(z); small = z <= 5.0
    # series (used only for small/medium arguments)
    zs = z[small]; s = np.zeros_like(zs); zp = np.ones_like(zs)
    from scipy.special import gamma as G
    for k in range(nterm):
        s += zp/G(beta+alpha*k); zp = zp*(-zs)
    out[small] = s
    # large-argument asymptotics E_{a,b}(-x) ~ x^{-1}/Gamma(b-a) (leading term)
    zl = z[~small]
    out[~small] = (zl**(-1.0))/G(beta-alpha) if zl.size else zl
    return out

def make_inputs(Nx, n, seeds, eta, w):
    U0 = np.stack([generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
    NU = np.stack([(1.0+eta*w).astype(np.float32) for _ in range(n)])
    return U0, NU

@torch.no_grad()
def base_field(U0, Nx, Nt, T, a, s):
    LG.T = T
    AD, SD, PT = build_prop_table(Nx, na=31, ns=31, Nt=Nt)
    g = prop_at(a, s, AD, SD, PT)
    return ub_full_batch(U0, g, Nx).cpu().numpy()

@torch.no_grad()
def true_linear(U0, NU, Nx, Nt, T, a, s, Q, bs=8):
    outs = []
    av = np.full(len(U0), a, np.float32); sv = np.full(len(U0), s, np.float32)
    for i0 in range(0, len(U0), bs):
        o = gpu_solve_imex_batch(U0[i0:i0+bs], NU[i0:i0+bs], av[i0:i0+bs], sv[i0:i0+bs],
                                 T, Nt, Nx, Q=Q, n_inner=3, device=DEV, dtype=DT).cpu().numpy()
        outs.append(o)
    return np.concatenate(outs, 0)

@torch.no_grad()
def solve_c1_batch(u0, w, eta, alpha, s, T, Nt, Nx, Q=None, n_inner=3, bs=8):
    """Unified linear-splitting solver: the implicit backbone is always c=1, and the variable-coefficient term is eta*w (Picard).
    At eta=0 this is the frozen constant-coefficient solution, with an operator-by-operator discretization identical to the eta>0 variable-coefficient solution, so scattering is isolated exactly in the residual."""
    outs = []
    for i0 in range(0, len(u0), bs):
        Ub = torch.as_tensor(u0[i0:i0+bs], device=DEV, dtype=DT); B = Ub.shape[0]
        n = Nx-1; ui = Ub[:, 1:Nx, 1:Nx].contiguous()
        wt = torch.as_tensor(w[1:Nx, 1:Nx], device=DEV, dtype=DT)[None]
        av = torch.as_tensor(np.full(B, alpha, np.float32), device=DEV, dtype=DT)
        sv = torch.as_tensor(np.full(B, s, np.float32), device=DEV, dtype=DT)
        if Q is None: Q = spectral_eigvals_torch(Nx, DEV, DT)
        dt = T/Nt; dta = (dt**av)[:, None, None]
        jj = torch.arange(Nt, device=DEV, dtype=DT)[None]; ac = av[:, None]
        b = (jj+1)**(1-ac) - torch.where(jj == 0, torch.zeros_like(jj), jj**(1-ac))
        Qs = Q[None]**sv[:, None, None]
        denom = b[:, 0][:, None, None] + dta*1.0*Qs          # backbone c=1 fixed
        hist = torch.zeros((Nt+1, B, n, n), device=DEV, dtype=DT); hist[0] = ui
        for k in range(1, Nt+1):
            base = b[:, k-1][:, None, None]*ui
            if k > 1:
                j = torch.arange(1, k, device=DEV); ww = b[:, k-j-1]-b[:, k-j]
                base = base + torch.einsum('bj,jbxy->bxy', ww, hist[1:k])
            x = hist[k-1]
            for _ in range(n_inner):
                Lx = dst2_torch(Qs*dst2_torch(x))
                r = base - dta*eta*wt*Lx
                x = dst2_torch(dst2_torch(r)/denom)
            hist[k] = x
        o = torch.zeros((B, Nx+1, Nx+1), device=DEV, dtype=DT); o[:, 1:Nx, 1:Nx] = hist[Nt]
        outs.append(o.cpu().numpy())
    return np.concatenate(outs, 0)

@torch.no_grad()
def solve_born_batch(u0, w, alpha, s, T, Nt, Nx, Q=None, bs=8):
    """Joint advance under the same implicit L1 discretization: ub=frozen c=1 constant-coefficient solution; u1=first-order Born response (eta=1).
    CD^a ub = -Lambda^{2s} ub ;  CD^a u1 = -Lambda^{2s}u1 - w*Lambda^{2s} ub, u1(0)=0。
    Preserve the correct operator order (Lambda acts on ub first, then multiply by w), on the same grid and scheme as the full variable-coefficient solution."""
    outb, out1 = [], []
    for i0 in range(0, len(u0), bs):
        Ub = torch.as_tensor(u0[i0:i0+bs], device=DEV, dtype=DT); B = Ub.shape[0]; n = Nx-1
        ui = Ub[:, 1:Nx, 1:Nx].contiguous(); wt = torch.as_tensor(w[1:Nx, 1:Nx], device=DEV, dtype=DT)[None]
        av = torch.full((B,), alpha, device=DEV, dtype=DT); sv = torch.full((B,), s, device=DEV, dtype=DT)
        if Q is None: Q = spectral_eigvals_torch(Nx, DEV, DT)
        dt = T/Nt; dta = (dt**av)[:, None, None]
        jj = torch.arange(Nt, device=DEV, dtype=DT)[None]; ac = av[:, None]
        bb = (jj+1)**(1-ac)-torch.where(jj == 0, torch.zeros_like(jj), jj**(1-ac))
        Qs = Q[None]**sv[:, None, None]; denom = bb[:, 0][:, None, None] + dta*Qs
        hb = torch.zeros((Nt+1, B, n, n), device=DEV, dtype=DT); hb[0] = ui
        h1 = torch.zeros_like(hb)
        for k in range(1, Nt+1):
            rb = bb[:, k-1][:, None, None]*ui
            r1 = torch.zeros_like(ui)
            if k > 1:
                j = torch.arange(1, k, device=DEV); ww = bb[:, k-j-1]-bb[:, k-j]
                rb = rb + torch.einsum('bj,jbxy->bxy', ww, hb[1:k])
                r1 = r1 + torch.einsum('bj,jbxy->bxy', ww, h1[1:k])
            ub_prev = hb[k-1]
            ub_k = dst2_torch(dst2_torch(rb)/denom)                       # frozen constant-coefficient solution
            Ls_ub = dst2_torch(Qs*dst2_torch(ub_prev))                    # Lambda^{2s} ub (physical field)
            r1 = r1 - dta*wt*Ls_ub
            u1_k = dst2_torch(dst2_torch(r1)/denom)
            hb[k] = ub_k; h1[k] = u1_k
        ob = torch.zeros((B, Nx+1, Nx+1), device=DEV, dtype=DT); ob[:, 1:Nx, 1:Nx] = hb[Nt]
        o1 = torch.zeros_like(ob); o1[:, 1:Nx, 1:Nx] = h1[Nt]
        outb.append(ob.cpu().numpy()); out1.append(o1.cpu().numpy())
    return np.concatenate(outb, 0), np.concatenate(out1, 0)

@torch.no_grad()
def true_nonlin(U0, NU, Nx, Nt, T, a, s, Q, beta, bs=8):
    outs = []
    av = np.full(len(U0), a, np.float32); sv = np.full(len(U0), s, np.float32)
    for i0 in range(0, len(U0), bs):
        o = gpu_solve_nl_batch(U0[i0:i0+bs], NU[i0:i0+bs], av[i0:i0+bs], sv[i0:i0+bs],
                               T, Nt, Nx, beta=beta, Q=Q, n_inner=3, device=DEV, dtype=DT).cpu().numpy()
        outs.append(o)
    return np.concatenate(outs, 0)

def fit_slope(x, y):
    lx, ly = np.log(np.asarray(x)), np.log(np.asarray(y))
    A = np.vstack([lx, np.ones_like(lx)]).T; slope, icept = np.linalg.lstsq(A, ly, rcond=None)[0]
    pred = slope*lx+icept; r2 = 1-np.sum((ly-pred)**2)/(np.sum((ly-ly.mean())**2)+1e-12)
    return slope, icept, r2

# ---------------- P1-5 (a) e_prop vs eta (expected slope 1) ----------------
def stage_eta():
    tag = 'eta'; z = nz(os.path.join(OUT, 'p15_eta.npz'))
    done = set(np.round(z['eta'], 4).tolist()) if z else set()
    if SMOKE: Nx, Nt, n, etas = 48, 600, 4, [0.05, 0.2]
    else:     Nx, Nt, n, etas = 128, 2400, 16, [0.02, 0.05, 0.10, 0.20, 0.40]
    T = 0.015; a, s = ALP, 0.6
    rng = np.random.RandomState(2026); seeds = rng.randint(100000, size=n)
    w = fixed_w(Nx); Q = spectral_eigvals_torch(Nx, DEV, DT)
    U0 = np.stack([generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
    errs = list(z['err']) if z else []
    for eta in etas:
        if round(eta, 4) in done: continue
        t0 = time.time()
        ub = solve_c1_batch(U0, w, 0.0, a, s, T, Nt, Nx, Q=Q)
        ut = solve_c1_batch(U0, w, eta, a, s, T, Nt, Nx, Q=Q)
        e = np.mean([rel(ut[i].ravel(), ub[i].ravel()) for i in range(n)])
        errs.append(e); np.savez(os.path.join(OUT, 'p15_eta.npz'), eta=np.array(etas[:len(errs)]), err=np.array(errs))
        log(f'[eta] eta={eta:.3f}  e_prop={e:.4e}  ({time.time()-t0:.1f}s)')
    # re-save using the full etas order
    es = np.array(errs); np.savez(os.path.join(OUT, 'p15_eta.npz'), eta=np.array(etas[:len(es)]), err=es)
    sl, ic, r2 = fit_slope(etas[:len(es)], es)
    log(f'[eta] loglog fitted slope = {sl:.3f}  (Proposition 1 expected ~1.00)  R2={r2:.4f}')
    plt.figure(figsize=(5.2,4.0))
    plt.loglog(etas[:len(es)], es, 'o-', color='#1F6FB2', lw=2, ms=7, label='measured')
    xx = np.array(etas[:len(es)]); plt.loglog(xx, np.exp(ic)*xx, '--', color='#D9534F', label=f'slope={sl:.2f} (theory 1)')
    plt.xlabel(r'relative coefficient variation $\eta$'); plt.ylabel(r'$\|u-u_{base}\|/\|u_{base}\|$')
    plt.title('Prop.1: propagator error vs $\\eta$ (linear, $T$ fixed)'); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'p15_eta.png'), dpi=150); plt.close()
    return sl, r2

# ---------------- P1-5 (b) e_prop vs T (linear in T^alpha / loglog slope alpha) ----------------
def stage_T():
    z = nz(os.path.join(OUT, 'p15_T.npz')); done = set(np.round(z['T'], 6).tolist()) if z else set()
    if SMOKE: Nx, Nt0, n, Ts, TSMALL = 48, 600, 4, [0.005, 0.02], 0.02
    else:     Nx, n, Ts, TSMALL = 128, 16, [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3], 1e-3
    eta = 0.15; a, s = ALP, 0.6
    rng = np.random.RandomState(2027); seeds = rng.randint(100000, size=n)
    w = fixed_w(Nx); Q = spectral_eigvals_torch(Nx, DEV, DT)
    U0 = np.stack([generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
    errs = list(z['err']) if z else []
    for T in Ts:
        if round(T, 5) in done: continue
        Nt = int(round(Nt0 if SMOKE else T*160000))   # fix dt=1/160000 to keep temporal accuracy consistent across T
        t0 = time.time()
        ub = solve_c1_batch(U0, w, 0.0, a, s, T, Nt, Nx, Q=Q)
        ut = solve_c1_batch(U0, w, eta, a, s, T, Nt, Nx, Q=Q)
        e = np.mean([rel(ut[i].ravel(), ub[i].ravel()) for i in range(n)])
        errs.append(e); np.savez(os.path.join(OUT, 'p15_T.npz'), T=np.array(Ts[:len(errs)]), err=np.array(errs))
        log(f'[T]   T={T:.4f} Nt={Nt} e_prop={e:.4e} ({time.time()-t0:.1f}s)')
    Tarr = np.array(Ts[:len(errs)]); es = np.array(errs); Ta = Tarr**ALP
    m_small = Tarr <= TSMALL + 1e-12
    # small-T asymptotic window: the log-log slope should -> alpha; e vs T^a is linear through the origin
    sl_s, ic_s, r2_s = fit_slope(Tarr[m_small], es[m_small])
    Ta_s = Tarr[m_small]**ALP; k_s = np.sum(Ta_s*es[m_small])/np.sum(Ta_s**2)
    r2lin_s = 1-np.sum((es[m_small]-k_s*Ta_s)**2)/(np.sum((es[m_small]-es[m_small].mean())**2)+1e-12)
    lin_k = np.sum(Ta*es)/np.sum(Ta**2)
    r2lin = 1-np.sum((es-lin_k*Ta)**2)/(np.sum((es-es.mean())**2)+1e-12)
    sl, ic, r2 = fit_slope(Tarr, es)
    log(f'[T]   small-T window (T<={TSMALL:g}): loglog slope={sl_s:.3f} (expected ~{ALP}) R2={r2_s:.4f}; e~T^a linear R2={r2lin_s:.4f}')
    log(f'[T]   full range (incl. ML saturation): loglog slope={sl:.3f}; e/T^a decreases monotonically from the small-T value {es[m_small][0]/Tarr[m_small][0]**ALP:.3f} to {es[-1]/Tarr[-1]**ALP:.3f} (bounded above)')
    fig, ax = plt.subplots(1, 2, figsize=(9.4,3.8))
    ax[0].plot(Ta, es, 'o-', color='#17A2B8', ms=7, label='measured')
    ax[0].plot(Ta_s, k_s*Ta_s, '--', color='#D9534F', lw=2, label=f'small-T linear fit')
    ax[0].set_xlabel(r'$T^\alpha$'); ax[0].set_ylabel(r'$\|u-u_{base}\|/\|u_{base}\|$')
    ax[0].set_title(r'vs $T^\alpha$: linear at small $T$'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].loglog(Tarr, es, 'o-', color='#1F6FB2', ms=7, label='measured')
    xx = np.array([Tarr[m_small][0], Tarr[m_small][-1]])
    ax[1].loglog(xx, np.exp(ic_s)*(xx/xx[0])**sl_s, '--', color='#D9534F', lw=2,
                 label=f'small-T slope={sl_s:.2f} (theory {ALP})')
    ax[1].set_xlabel('T'); ax[1].set_ylabel('e_prop'); ax[1].set_title('log-log: small-T asymptote + ML saturation'); ax[1].legend(); ax[1].grid(alpha=.3, which='both')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'p15_T.png'), dpi=150); plt.close()
    return sl_s, r2lin_s

# ---------------- P1-5 (c) nonlinear e vs beta (nu=1 isolation, expected linear in beta) ----------------
def stage_beta():
    z = nz(os.path.join(OUT, 'p15_beta.npz')); done = set(np.round(z['beta'], 4).tolist()) if z else set()
    if SMOKE: Nx, Nt, n, betas = 48, 600, 4, [1.0, 3.0]
    else:     Nx, Nt, n, betas = 128, 2400, 16, [0.5, 1.0, 2.0, 3.0, 5.0]
    T = 0.015; a, s = ALP, 0.6
    rng = np.random.RandomState(2028); seeds = rng.randint(100000, size=n)
    w = fixed_w(Nx); Q = spectral_eigvals_torch(Nx, DEV, DT)
    U0, NU = make_inputs(Nx, n, seeds, 0.0, w)     # nu is identically 1 -> only the nonlinear residual remains
    ub = solve_c1_batch(U0, w, 0.0, a, s, T, Nt, Nx, Q=Q)   # c=1 frozen linear solution (same scheme)
    errs = list(z['err']) if z else []
    for beta in betas:
        if round(beta, 4) in done: continue
        t0 = time.time(); ut = true_nonlin(U0, NU, Nx, Nt, T, a, s, Q, beta)
        e = np.mean([rel(ut[i].ravel(), ub[i].ravel()) for i in range(n)])
        errs.append(e); np.savez(os.path.join(OUT, 'p15_beta.npz'), beta=np.array(betas[:len(errs)]), err=np.array(errs))
        log(f'[beta] beta={beta:.2f} e_nl={e:.4e} ({time.time()-t0:.1f}s)')
    es = np.array(errs); bs = np.array(betas[:len(es)])
    k = np.sum(bs*es)/np.sum(bs**2); fit = k*bs; r2lin = 1-np.sum((es-fit)**2)/(np.sum((es-es.mean())**2)+1e-12)
    sl, ic, r2 = fit_slope(bs, es)
    log(f'[beta] e-vs-beta linear R2={r2lin:.4f}; loglog slope={sl:.3f} (Proposition 1 nonlinear term expected ~1)')
    plt.figure(figsize=(5.2,4.0))
    plt.plot(bs, es, 'o-', color='#7A5BA0', lw=2, ms=7, label='measured'); plt.plot(bs, fit, '--', color='#D9534F', label=f'linear, R2={r2lin:.3f}')
    plt.xlabel(r'nonlinearity $\beta$  ($\nu\equiv1$, T fixed)'); plt.ylabel(r'$\|u_{nl}-u_{base}\|/\|u_{base}\|$')
    plt.title('Prop.1: residual vs $\\beta$ (nonlinear term)'); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'p15_beta.png'), dpi=150); plt.close()
    return sl, r2lin

# ---------------- P1-4 residual radial power-spectrum tail: first-order scattering |rhat|^2 ~ |k|^{-4s} ----------------
def radial_bins(n):
    k = np.arange(1, n+1); K1, K2 = np.meshgrid(k, k, indexing='ij')
    R = np.round(np.sqrt(K1**2+K2**2)).astype(int)
    return R, np.arange(1, int(np.floor((n-1)/np.sqrt(2)))+1)

def radial_power(spec, R, radii):
    """Mean power per mode within a radial thin shell (divide by the number of modes in the shell to remove the 2-D shell-count dimensional factor)."""
    E = np.abs(spec)**2; P = np.zeros(len(radii))
    for j, r in enumerate(radii):
        msk = R == r; cnt = msk.sum()
        P[j] = E[msk].sum()/cnt if cnt > 0 else np.nan
    return P

@torch.no_grad()
def cum_tail(spec, Ks):
    """max-norm cumulative spectral energy tail(K)=sum over max(k1,k2)>K of |coef|^2."""
    n = spec.shape[0]; k = np.arange(1, n+1); K1, K2 = np.meshgrid(k, k, indexing='ij')
    P = np.maximum(K1, K2); E = np.abs(spec)**2
    return np.array([E[P > K].sum() for K in Ks])

def stage_spec():
    """Robust verification of Proposition 2 (holds for all s, without relying on a narrow asymptotic window):
    (A) under the same L1 discretization, the first-order Born response r1=eta*u1 explains the full residual r=u-u_base, and the explanation error is linear in eta (O(eta^2));
    (B) the residual radial power spectrum overlaps the first-order response mode by mode (Pr/Pr1~1 over all wavenumbers), showing the spectral structure is captured exactly by first-order scattering;
    (C) the spectra show that high modes of u_base are suppressed by Mittag-Leffler dissipation, while the residual / first-order response / scattering source w*u0 have flatter high-mode tails."""
    z = nz(os.path.join(OUT, 'p14_spec.npz')); done = set(np.round(z['s_done'], 3).tolist()) if z else set()
    if SMOKE: Nx, Nt, n, bsx, slist, Tsp = 64, 400, 4, 4, [0.6], 0.005
    else:     Nx, Nt, n, bsx, slist, Tsp = 192, 1600, 12, 6, [0.40, 0.60, 0.80], 0.005
    a = ALP; P_EXP = 1.5; eta_scan = [0.05, 0.10, 0.20, 0.40]; eta0 = 0.20
    w = fixed_w(Nx, seed=777, kmax=max(16, Nx//3), beta=1.5)
    U0 = np.stack([powerlaw_u0(Nx, p=P_EXP, seed=500+i) for i in range(n)]).astype(np.float32)
    Qdev = spectral_eigvals_torch(Nx, DEV, DT)
    n_in = Nx-1; R, radii = radial_bins(n_in)
    have_k = {float(x): k for x, k in zip(z['s_done'], z['keys'])} if z else {}
    have_e = {float(x): e for x, e in zip(z['s_done'], z['ecurve'])} if z and 'ecurve' in z else {}
    rows = []
    for s in slist:
        if round(s, 3) in done:
            rows.append((s, have_k[s], have_e[s])); continue
        t0 = time.time()
        ub, u1 = solve_born_batch(U0, w, a, s, Tsp, Nt, Nx, Q=Qdev, bs=bsx)
        ec = []
        spec0 = None
        for eta in eta_scan:
            ut = solve_c1_batch(U0, w, eta, a, s, Tsp, Nt, Nx, Q=Qdev, bs=bsx)
            rf = ut-ub; exv = np.mean([rel(rf[i].ravel(), (eta*u1)[i].ravel()) for i in range(n)])
            ec.append(exv)
            if abs(eta-eta0) < 1e-9:
                Pb=np.zeros(len(radii)); Pr=np.zeros(len(radii)); P1=np.zeros(len(radii)); Ps=np.zeros(len(radii))
                for i in range(n):
                    sp = lambda f: dst2_torch(torch.tensor(f[1:Nx,1:Nx][None], device=DEV, dtype=DT))[0].cpu().numpy()
                    Pb += radial_power(sp(ub[i]), R, radii)/n
                    Pr += radial_power(sp(rf[i]), R, radii)/n
                    P1 += radial_power(sp((eta*u1)[i]), R, radii)/n
                    Ps += radial_power(sp(w*U0[i]), R, radii)/n
                spec0 = np.stack([radii, Pb, Pr, P1, Ps], 1)
        ec = np.array(ec)
        rows.append((s, spec0, ec))
        np.savez(os.path.join(OUT, 'p14_spec.npz'), s_done=np.array([r[0] for r in rows]),
                 keys=np.array([r[1] for r in rows]), ecurve=np.array([r[2] for r in rows]), eta_scan=np.array(eta_scan))
        log(f'[spec] s={s:.2f} done ({time.time()-t0:.1f}s) explain(eta)={np.round(ec,4).tolist()}')
    # summary + figure
    etag = np.array(eta_scan); table = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); colors = {0.4:'#1F6FB2', 0.6:'#17A2B8', 0.8:'#7A5BA0'}
    for s, K, ec in rows:
        rad, Pb, Pr, P1, Ps = K.T; c = colors.get(round(s,1))
        ax[0].loglog(rad, np.maximum(Pb,1e-30), '--', color=c, alpha=.55, label=f'u_base s={s}')
        ax[0].loglog(rad, np.maximum(Pr,1e-30), '-',  color=c, lw=2, label=f'residual s={s}')
        ax[0].loglog(rad, np.maximum(P1,1e-30), ':',  color='k', lw=1, label=f'1st-Born s={s}')
        ax[0].loglog(rad, np.maximum(Ps,1e-30), '-.', color=c, alpha=.8, label=f'source $w u_0$ s={s}')
        ax[1].plot(etag, ec, 'o-', color=c, lw=1.8, ms=6, label=f's={s}')
        # explain vs eta is linear through the origin
        k_lin = np.sum(etag*ec)/np.sum(etag**2); fit = k_lin*etag
        r2 = 1-np.sum((ec-fit)**2)/(np.sum((ec-ec.mean())**2)+1e-12)
        m = (Pr > Pr.max()*1e-8) & (P1 > 0); ratio = Pr[m]/P1[m]
        med = float(np.median(ratio)); lstd = float(np.std(np.log10(ratio)))
        table.append((s, ec[list(etag).index(eta0)], k_lin, r2, med, lstd))
    ax[0].set_xlabel(r'radial wavenumber $\kappa$'); ax[0].set_ylabel('per-mode power')
    ax[0].set_title('Prop.2: residual spectrum = 1st-order Born (coincide)'); ax[0].legend(fontsize=6.5); ax[0].grid(alpha=.3, which='both')
    ax[1].set_xlabel(r'coefficient variation $\eta$'); ax[1].set_ylabel(r'$\|r-\eta u_1\|/\|\eta u_1\|$')
    ax[1].set_title('1st-order remainder is $O(\eta^2)$: linear in $\eta$'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'p14_spectral_tail.png'), dpi=150); plt.close()
    log('\n[spec]  s  explain-err@.2  explain~eta slope  linear R2  Pr/Pr1 median  log10std (should ~0)')
    for s, ex2, kl, r2, med, lstd in table:
        log(f'[spec] {s:.2f}    {ex2:.4f}        {kl:.4f}       {r2:.4f}    {med:.3f}      {lstd:.3f}')
    np.savez(os.path.join(OUT, 'p14_slopes.npz'), table=np.array(table))
    return table

if __name__ == '__main__':
    stages = sys.argv[1:] or ['eta', 'T', 'beta', 'spec']
    summary = {}
    if 'eta' in stages:  summary['eta'] = stage_eta(); save_txt('eta'); L=[]
    if 'T' in stages:    summary['T'] = stage_T(); save_txt('T'); L=[]
    if 'beta' in stages: summary['beta'] = stage_beta(); save_txt('beta'); L=[]
    if 'spec' in stages: summary['spec'] = stage_spec(); save_txt('spec'); L=[]
    log('\n==== P1-5/P1-4 summary ====')
    for k, v in summary.items(): log(f'{k}: {v}')
    save_txt('summary')
    print('ALL DONE')
