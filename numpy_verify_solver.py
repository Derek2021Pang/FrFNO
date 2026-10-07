# -*- coding: utf-8 -*-
"""
Independent NumPy (float64) reference solver, written from the paper's Algorithm
(refsolver): DST-I spectral, L1 time stepping, IMEX/Picard, first-order upwind
advection. Cross-validates the GPU IMEX implementation
(nl_solver.gpu_solve_nl_batch) on the nonlinear fractional Burgers equation
^CD_t^a u + beta*u*d_x u = -nu(x,y)(-Delta)^s u, homogeneous Dirichlet.

No torch and no GPU solver code is used below; the two implementations share
only the mathematical discretization described in the paper.
"""
import time
import numpy as np
from scipy.fft import dst as _dst
from scipy.special import gamma as _gamma


def dst2(x):
    """2-D DST-I (ortho, self-inverse) of an (n,n) interior field."""
    return _dst(_dst(x, type=1, norm='ortho', axis=-1), type=1, norm='ortho', axis=-2)


def spectral_eigvals(Nx):
    """Five-point Dirichlet eigenvalues lambda_k = 4 sin^2(k pi h/2)/h^2 summed per axis."""
    h = 1.0 / Nx
    m = np.arange(1, Nx, dtype=np.float64)
    s = (4.0 / h ** 2) * np.sin(np.pi * h * m / 2.0) ** 2
    QX, QY = np.meshgrid(s, s, indexing='ij')
    return QX + QY


def l1_b(alpha, Nt):
    """Standard L1 weights b_j = [(j+1)^(1-a) - j^(1-a)] / Gamma(2-a)."""
    j = np.arange(Nt, dtype=np.float64)
    bp1 = (j + 1.0) ** (1.0 - alpha)
    bj = np.where(j == 0, 0.0, j ** (1.0 - alpha))
    return (bp1 - bj) / _gamma(2.0 - alpha)


def ddx_upwind(u, Nx):
    """First-order upwind d_x u with zero Dirichlet boundaries (velocity = u itself)."""
    n = u.shape[0]
    dx = 1.0 / Nx
    up = np.zeros((n + 2, n + 2))
    up[1:-1, 1:-1] = u
    uc = up[1:-1, 1:-1]
    back = (uc - up[:-2, 1:-1]) / dx
    fwd = (up[2:, 1:-1] - uc) / dx
    return np.where(uc >= 0, back, fwd)


def solve_numpy(u0, nu, alpha, s, T, Nt, Nx, beta=3.0, n_inner=2):
    """Solve one trajectory. u0, nu: full (Nx+1,Nx+1) grids with zero boundaries.
    Returns the full-grid solution at T."""
    n = Nx - 1
    Q = spectral_eigvals(Nx)
    Qs = Q ** s
    dt = T / Nt
    dta = dt ** alpha
    b = l1_b(alpha, Nt)
    ui = u0[1:Nx, 1:Nx].astype(np.float64)
    ni = nu[1:Nx, 1:Nx].astype(np.float64)
    c = ni.max()
    denom = b[0] + dta * c * Qs
    hist = np.zeros((Nt + 1, n, n))
    hist[0] = ui
    for k in range(1, Nt + 1):
        base = b[k - 1] * ui
        if k > 1:
            j = np.arange(1, k)
            ww = b[k - j - 1] - b[k - j]
            base = base + np.einsum('j,jxy->xy', ww, hist[1:k])
        adv = hist[k - 1] * ddx_upwind(hist[k - 1], Nx)
        r = base - dta * beta * adv
        x = hist[k - 1]
        for _ in range(n_inner):
            Lx = dst2(Qs * dst2(x))
            rr = r - dta * (ni - c) * Lx
            x = dst2(dst2(rr) / denom)
        hist[k] = x
    out = np.zeros((Nx + 1, Nx + 1))
    out[1:Nx, 1:Nx] = hist[Nt]
    return out


def make_fields(Nx, seed=7):
    """Deterministic smooth test fields (positive variable diffusivity, sign-changing u0)."""
    x = np.linspace(0.0, 1.0, Nx + 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y) * (1.0 + 0.4 * X + 0.2 * Y)
    nu = 1.0 + 0.5 * np.cos(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    return u0, nu


def main():
    import os, sys
    ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, ROOT)
    import torch
    from nl_solver import gpu_solve_nl_batch

    lines = []
    lines.append("Cross-validation: independent NumPy (float64) vs GPU IMEX (float64)")
    lines.append("Equation: ^CD_t^a u + beta u d_x u = -nu(x,y)(-Delta)^s u, Dirichlet")
    configs = [
        (16, 400, 0.75, 0.75, 3.0, 0.015),
        (16, 400, 0.60, 0.85, 0.0, 0.015),   # beta=0: pure variable-coefficient diffusion
        (32, 800, 0.85, 0.60, 3.0, 0.020),
        (32, 800, 1.00, 1.00, 3.0, 0.015),   # integer order limit
    ]
    worst = 0.0
    for (Nx, Nt, a, s, beta, T) in configs:
        u0, nu = make_fields(Nx)
        t0 = time.time()
        u_np = solve_numpy(u0, nu, a, s, T, Nt, Nx, beta=beta)
        t_np = time.time() - t0
        u0b = torch.tensor(u0[None], dtype=torch.float64, device='cuda')
        nub = torch.tensor(nu[None], dtype=torch.float64, device='cuda')
        t0 = time.time()
        u_gpu = gpu_solve_nl_batch(u0b, nub, np.array([a]), np.array([s]),
                                   T, Nt, Nx, beta=beta, device='cuda',
                                   dtype=torch.float64)[0].cpu().numpy()
        t_gpu = time.time() - t0
        diff = np.abs(u_np - u_gpu).max()
        rel = diff / (np.abs(u_np).max() + 1e-30)
        worst = max(worst, diff)
        lines.append(
            "Nx=%d Nt=%d alpha=%.2f s=%.2f beta=%.1f T=%.3f:  max|d| = %.6e   rel = %.6e   (np %.2fs, gpu %.2fs)"
            % (Nx, Nt, a, s, beta, T, diff, rel, t_np, t_gpu))
    lines.append("worst max|diff| = %.6e (float64 machine precision ~2.2e-16 x #ops)" % worst)
    out = os.path.join(ROOT, "numpy_verify_result.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
