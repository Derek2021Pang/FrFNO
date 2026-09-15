# -*- coding: utf-8 -*-
"""
Nonlinear fractional Burgers (2-D scalar, advection in x) batched GPU solver.
Equation: ^CD_t^alpha u + beta*u*d_x u = -nu(x,y)(-Delta)^s u, homogeneous Dirichlet.
- Fractional diffusion reuses gpu_solver's DST-I spectral method, L1 time stepping, and the variable-coefficient IMEX/Picard fully implicit backbone;
- The nonlinear advection beta*u*d_x u is explicit (previous time level); d_x uses centered differences with zero Dirichlet boundary values;
  T is small and dt is tiny, hence CFL = max|u|*dt/dx << 1 and the explicit advection is stable.
- At beta=0 it must agree with gpu_solve_imex_batch (used for cross-validation).
"""
import numpy as np
import torch
from gpu_solver import spectral_eigvals_torch, dst2_torch


def ddx_central(u_in, Nx):
    """u_in: (B,n,n) interior field; returns the d_x u interior field with boundaries treated as 0, using second-order centered differences."""
    B, n, _ = u_in.shape
    dx = 1.0 / Nx
    up = torch.zeros((B, n + 2, n + 2), device=u_in.device, dtype=u_in.dtype)
    up[:, 1:-1, 1:-1] = u_in
    du = (up[:, 2:, 1:-1] - up[:, :-2, 1:-1]) / (2 * dx)
    return du


def ddx_upwind(u_in, Nx):
    """First-order upwind difference (transport velocity = u itself), zero boundaries; suppresses high-mode grid oscillations when fractional dissipation is weak at low s."""
    B, n, _ = u_in.shape
    dx = 1.0 / Nx
    up = torch.zeros((B, n + 2, n + 2), device=u_in.device, dtype=u_in.dtype)
    up[:, 1:-1, 1:-1] = u_in
    uc = up[:, 1:-1, 1:-1]
    back = (uc - up[:, :-2, 1:-1]) / dx          # (u_i - u_{i-1})/dx
    fwd = (up[:, 2:, 1:-1] - uc) / dx            # (u_{i+1} - u_i)/dx
    return torch.where(uc >= 0, back, fwd)


def gpu_solve_nl_batch(u0, nu, alpha, s, T, Nt, Nx, beta=3.0, Q=None,
                       device='cuda', dtype=torch.float32, n_inner=2, return_hist=False,
                       snap_idx=None):
    B = len(alpha); n = Nx - 1
    if Q is None:
        Q = spectral_eigvals_torch(Nx, device, dtype)
    dt = T / Nt
    alpha = torch.as_tensor(np.asarray(alpha), device=device, dtype=dtype)
    s = torch.as_tensor(np.asarray(s), device=device, dtype=dtype)
    dta = (dt ** alpha)[:, None, None]
    jj = torch.arange(Nt, device=device, dtype=dtype)[None]
    ac = alpha[:, None]
    bp1 = (jj + 1.0) ** (1.0 - ac)
    bj = torch.where(jj == 0, torch.zeros_like(jj), jj ** (1.0 - ac))
    b = bp1 - bj
    Qs = Q[None] ** s[:, None, None]
    ui = torch.as_tensor(u0[:, 1:Nx, 1:Nx], device=device, dtype=dtype)
    ni = torch.as_tensor(nu[:, 1:Nx, 1:Nx], device=device, dtype=dtype)
    c = ni.amax(dim=(1, 2), keepdim=True)
    hist = torch.zeros((Nt + 1, B, n, n), device=device, dtype=dtype)
    hist[0] = ui
    b0 = b[:, 0][:, None, None]
    denom = b0 + dta * c * Qs
    for k in range(1, Nt + 1):
        base = b[:, k - 1][:, None, None] * ui
        if k > 1:
            j = torch.arange(1, k, device=device)
            ww = b[:, k - j - 1] - b[:, k - j]
            base = base + torch.einsum('bj,jbxy->bxy', ww, hist[1:k])
        if k == Nt:
            last_base = base.clone()                       # L1 history weighting term (feeds the PINO residual)
        adv = hist[k - 1] * ddx_upwind(hist[k - 1], Nx)       # explicitnonlinearadvection u*d_xu(upwind)
        r = base - dta * beta * adv
        x = hist[k - 1]
        for _ in range(n_inner):
            Lx = dst2_torch(Qs * dst2_torch(x))
            rr = r - dta * (ni - c) * Lx
            x = dst2_torch(dst2_torch(rr) / denom)
        hist[k] = x
    out = torch.zeros((B, Nx + 1, Nx + 1), device=device, dtype=dtype)
    out[:, 1:Nx, 1:Nx] = hist[Nt]
    rets = [out]
    if return_hist:
        hb = torch.zeros((B, Nx + 1, Nx + 1), device=device, dtype=dtype)
        hb[:, 1:Nx, 1:Nx] = last_base
        rets.append(hb)
    if snap_idx is not None:
        SN = torch.zeros((len(snap_idx), B, Nx + 1, Nx + 1), device=device, dtype=dtype)
        for m, kk in enumerate(snap_idx):
            SN[m, :, 1:Nx, 1:Nx] = hist[kk]
        rets.append(SN)
    return rets[0] if len(rets) == 1 else tuple(rets)
