# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
Component-wise Method of Manufactured Solutions for the reference nonlinear
solver (constant nu=c0; manufactured u_e = sin(p pi x)sin(q pi y)(a0+b0 t^2)).

To stop one truncation error masking another, the manufactured source is built
component by component: the operator UNDER TEST is kept continuous (so its
discretization error survives), every other operator is injected exactly as the
numerical scheme sees it (so its error cancels algebraically).

 * TEMPORAL test: diffusion uses the DISCRETE 5-point eigenvalue on the mode
   (exact on the grid sine -> no spatial floor), Caputo uses the CONTINUOUS
   formula (L1 error survives), beta=0.  Expect order 2-alpha.
 * SPATIAL test: Caputo uses the DISCRETE L1 operator applied to the exact
   trajectory (cancels the temporal error exactly), diffusion uses the
   CONTINUOUS eigenvalue and advection the CONTINUOUS phi_x (so the 5-point O(h^2)
   / upwind O(h) errors survive, with NO temporal floor).
"""
import numpy as np
import torch
from scipy.special import gamma as G
from gpu_solver import spectral_eigvals_torch, dst2_torch
from nl_solver import ddx_upwind

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float64
c0 = 1.0
p, q = 1, 1
lam_c = np.pi**2*(p**2+q**2)          # continuous Dirichlet eigenvalue
a0, b0 = 0.5, 1.0


def qN(Nx, m):
    h = 1.0/Nx
    return (4.0/h**2)*np.sin(np.pi*h*m/2.0)**2


def fields(Nx, t):
    h = 1.0/Nx
    x = np.arange(Nx+1)*h
    X, Y = np.meshgrid(x, x, indexing='ij')
    phi = np.sin(p*np.pi*X)*np.sin(q*np.pi*Y)
    phix = p*np.pi*np.cos(p*np.pi*X)*np.sin(q*np.pi*Y)
    g = a0+b0*t**2
    return phi[1:Nx, 1:Nx], phix[1:Nx, 1:Nx], g


def cap_cont(alpha, t):
    # Standard continuous Caputo derivative: ^C D_t^alpha t^2 = Gamma(3)/Gamma(3-alpha) t^{2-alpha}.
    return 2.0/G(3.0-alpha)*t**(2.0-alpha)


def l1_weights(Nt, alpha):
    """Standard L1 weights b_j=[(j+1)^(1-alpha)-j^(1-alpha)]/Gamma(2-alpha)."""
    j = torch.arange(Nt, device=dev, dtype=DT)
    bp1 = (j+1.0)**(1.0-alpha)
    bj = torch.where(j == 0, torch.zeros_like(j), j**(1.0-alpha))
    return (bp1-bj) / G(2.0-alpha)


def run_temporal(Nx, Nt, alpha, s, T):
    """L1 temporal order; discrete diffusion (exact on mode), no advection."""
    Qs = spectral_eigvals_torch(Nx, dev, DT)**s
    dt = T/Nt; dta = dt**alpha
    bw = l1_weights(Nt, alpha)
    n = Nx-1
    phi0, _, g0 = fields(Nx, 0.0)
    hist = torch.zeros((Nt+1, 1, n, n), device=dev, dtype=DT)
    hist[0] = torch.tensor(phi0*g0, device=dev, dtype=DT)[None]
    for k in range(1, Nt+1):
        base = bw[k-1]*hist[0]
        if k > 1:
            jj = torch.arange(1, k, device=dev)
            base = base+torch.einsum('j,jbxy->bxy', bw[k-jj-1]-bw[k-jj], hist[1:k])
        phik, _, gk = fields(Nx, k*dt)
        phik = torch.tensor(phik, device=dev, dtype=DT)[None]
        Ldisc_ue = dst2_torch(Qs*dst2_torch(phik))          # discrete diffusion on mode
        f = phik*cap_cont(alpha, k*dt) + c0*gk*Ldisc_ue
        r = base+dta*f
        hist[k] = dst2_torch(dst2_torch(r)/(bw[0]+dta*c0*Qs))
    ue = torch.tensor(fields(Nx, T)[0]*fields(Nx, T)[2], device=dev, dtype=DT)
    return float(torch.linalg.norm(hist[Nt][0]-ue)/torch.linalg.norm(ue))


def run_spatial(Nx, Nt, alpha, s, beta, T):
    """Spatial order; discrete-L1 on the exact trajectory (no time floor),
    continuous diffusion eigenvalue and continuous advection."""
    Qs = spectral_eigvals_torch(Nx, dev, DT)**s
    dt = T/Nt; dta = dt**alpha
    bw = l1_weights(Nt, alpha)
    n = Nx-1
    phi0, phix0, g0 = fields(Nx, 0.0)
    ue_hist = [torch.tensor(phi0*g0, device=dev, dtype=DT)[None]]
    hist = torch.zeros((Nt+1, 1, n, n), device=dev, dtype=DT)
    hist[0] = ue_hist[0].clone()
    for k in range(1, Nt+1):
        tk = k*dt
        # discrete L1 of the EXACT trajectory
        base_e = bw[k-1]*ue_hist[0]
        for j in range(1, k):
            base_e = base_e+(bw[k-j-1]-bw[k-j])*ue_hist[j]
        phik, phixk, gk = fields(Nx, tk)
        uek = torch.tensor(phik*gk, device=dev, dtype=DT)[None]
        ue_hist.append(uek)
        L1disc = (bw[0]*uek-base_e)/dta
        adv_c = (gk**2)*torch.tensor(phik*phixk, device=dev, dtype=DT)[None]
        f = L1disc + c0*lam_c**s*uek + beta*adv_c
        # numerical advance
        base = bw[k-1]*hist[0]
        jj = torch.arange(1, k, device=dev)
        base = base+torch.einsum('j,jbxy->bxy', bw[k-jj-1]-bw[k-jj], hist[1:k])
        adv_d = hist[k-1]*ddx_upwind(hist[k-1], Nx)
        r = base-dta*beta*adv_d+dta*f
        hist[k] = dst2_torch(dst2_torch(r)/(bw[0]+dta*c0*Qs))
    ue = torch.tensor(fields(Nx, T)[0]*fields(Nx, T)[2], device=dev, dtype=DT)
    return float(torch.linalg.norm(hist[Nt][0]-ue)/torch.linalg.norm(ue))


def run_full(Nx, Nt, alpha, s, beta, T):
    """Fully continuous source (all operators continuous): true total reference error."""
    Qs = spectral_eigvals_torch(Nx, dev, DT)**s
    dt = T/Nt; dta = dt**alpha
    bw = l1_weights(Nt, alpha)
    n = Nx-1
    phi0, _, g0 = fields(Nx, 0.0)
    hist = torch.zeros((Nt+1, 1, n, n), device=dev, dtype=DT)
    hist[0] = torch.tensor(phi0*g0, device=dev, dtype=DT)[None]
    for k in range(1, Nt+1):
        base = bw[k-1]*hist[0]
        jj = torch.arange(1, k, device=dev)
        base = base+torch.einsum('j,jbxy->bxy', bw[k-jj-1]-bw[k-jj], hist[1:k])
        phik, phixk, gk = fields(Nx, k*dt)
        fc = torch.tensor(phik*cap_cont(alpha, k*dt), device=dev, dtype=DT)[None]
        fc += c0*lam_c**s*gk*torch.tensor(phik, device=dev, dtype=DT)[None]
        fc += beta*gk**2*torch.tensor(phik*phixk, device=dev, dtype=DT)[None]
        adv = hist[k-1]*ddx_upwind(hist[k-1], Nx)
        r = base-dta*beta*adv+dta*fc
        hist[k] = dst2_torch(dst2_torch(r)/(bw[0]+dta*c0*Qs))
    ue = torch.tensor(fields(Nx, T)[0]*fields(Nx, T)[2], device=dev, dtype=DT)
    return float(torch.linalg.norm(hist[Nt][0]-ue)/torch.linalg.norm(ue))


def ords(e):
    return [np.log(e[i]/e[i+1])/np.log(2.0) for i in range(len(e)-1)]


L = []
def log(z=''):
    print(z); L.append(z)


log('Component-wise MMS code verification (device=%s)' % dev)

log('\n=== (a) TEMPORAL L1 order (discrete diffusion, beta=0, T=0.20, Nx=32); theory 2-alpha ===')
Nts = [16, 32, 64, 128, 256, 512]
for al in [0.6, 0.75, 1.0]:
    e = [run_temporal(32, Nt, al, 0.75, 0.20) for Nt in Nts]
    po = ords(e)
    log('alpha=%.2f (expect %.2f):' % (al, 2-al))
    for k, Nt in enumerate(Nts):
        log('   Nt=%3d relL2=%.3e%s' % (Nt, e[k], '' if k == 0 else '   order=%.3f' % po[k-1]))

log('\n=== (b) SPATIAL order, nonlinear upwind beta=3 (discrete L1, Nt=200, alpha=s=.75,T=.02); ->1 ===')
Nxs = [16, 32, 64, 128, 256]
eb = [run_spatial(Nx, 200, 0.75, 0.75, 3.0, 0.02) for Nx in Nxs]
pb = ords(eb)
for k, Nx in enumerate(Nxs):
    log('   Nx=%3d h=%.5f relL2=%.3e%s' % (Nx, 1/Nx, eb[k], '' if k == 0 else '   order=%.3f' % pb[k-1]))

log('\n=== (c) SPATIAL order, diffusion only beta=0 (discrete L1, Nt=200, alpha=s=.75,T=.02); ->2 ===')
ec = [run_spatial(Nx, 200, 0.75, 0.75, 0.0, 0.02) for Nx in Nxs]
pc = ords(ec)
for k, Nx in enumerate(Nxs):
    log('   Nx=%3d h=%.5f relL2=%.3e%s' % (Nx, 1/Nx, ec[k], '' if k == 0 else '   order=%.3f' % pc[k-1]))

log('\n=== production reference bound, fully continuous source, Nx=128,Nt=1600,beta=3 ===')
for TT in [0.015, 0.02]:
    log('   T=%.3f total MMS relL2=%.3e' % (TT, run_full(128, 1600, 0.75, 0.75, 3.0, TT)))

import os
ROOT = _PKG_ROOT
open(os.path.join(ROOT, 'mms_result.txt'), 'w', encoding='utf-8').write('\n'.join(L))
log('\nsaved mms_result.txt')
