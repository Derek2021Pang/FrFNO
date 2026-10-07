"""
Space-time two-orders fractional diffusion solver (rebuilt from the authoritative archived finite-difference formulas)
    ^C D_t^alpha u = -nu(x,y) (-Delta)^s u,  homogeneous Dirichlet,  u(x,0)=u0

- Space: spectral fractional Laplacian, diagonalized by DST-I (ortho); (-Delta)^s u = iDST[ q^s DST(u) ]
- Time: standard Caputo L1 scheme, b_j = [(j+1)^(1-alpha) - j^(1-alpha)] / Gamma(2-alpha)
    * constant-coefficient implicit (mode by mode; exact reference / analytic baseline):
        (b0 + dt^a lam) v_n = b[n-1] v0 + sum_{j=1}^{n-1}(b[n-j-1]-b[n-j]) v_j
    * variable-coefficient explicit (nu does not commute with the nonlocal operator, so it cannot be diagonalized):
        b0 u_n = b[n-1] u0 + sum_j(...) u_j - dt^a * nu * (-Delta)^s u_{n-1}
Grid: (Nx+1)^2 points, (Nx-1)^2 interior, with the outer ring as the zero boundary.
"""
import numpy as np
from scipy.fft import dst


def dst2(a):
    """Apply orthogonal DST-I along both spatial axes of an interior 2-D field (it is its own inverse)."""
    return dst(dst(a, type=1, axis=-2, norm='ortho'), type=1, axis=-1, norm='ortho')


def spectral_eigvals(Nx):
    """Interior eigenvalues of the five-point Laplacian, q_mn = (4/h^2)[sin^2(m*pi*h/2)+sin^2(n*pi*h/2)]."""
    h = 1.0 / Nx
    m = np.arange(1, Nx)
    s = (4.0 / h ** 2) * np.sin(np.pi * h * m / 2.0) ** 2
    QX, QY = np.meshgrid(s, s, indexing='ij')
    return QX + QY


def _l1_b(alpha, Nt):
    """Standard L1 coefficients b_j=[(j+1)^(1-alpha)-j^(1-alpha)]/Gamma(2-alpha), j=0..Nt-1.
    b0=1/Gamma(2-alpha); at alpha=1 it reduces to backward Euler (b0=1, b_j=0 for j>0).

    Note that at j=0 the term 0^(1-alpha) must be treated as 0; at alpha=1 numpy 0**0=1 would
    wrongly yield b0=0, so this case is separated explicitly.
    """
    from scipy.special import gamma as Gamma
    j = np.arange(Nt)
    bp1 = (j + 1.0) ** (1.0 - alpha)
    bj = np.zeros_like(j, dtype=float)
    m = j > 0
    bj[m] = j[m] ** (1.0 - alpha)              # avoid the 0-to-a-negative-power warning at j=0 when (1-alpha)<0
    return (bp1 - bj) / Gamma(2.0 - alpha)


def _to_inner(u0, Nx):
    return u0[1:Nx, 1:Nx] if u0.shape[0] == Nx + 1 else u0.copy()


def _wrap(inner, Nx):
    out = np.zeros((Nx + 1, Nx + 1), dtype=inner.dtype)
    out[1:Nx, 1:Nx] = inner
    return out


def frac_lap_const(u_inner, Qs):
    """Constant-order spectral fractional Laplacian (-Delta)^s u = iDST[q^s DST u] (Qs=Q**s)."""
    return dst2(Qs * dst2(u_inner))


def solve_const_implicit(u0, alpha, s, T, Nt, Nx, Q=None):
    """Constant-coefficient nu=1 implicit L1 (DST mode-by-mode scalar recursion), used as the exact reference."""
    if Q is None:
        Q = spectral_eigvals(Nx)
    ui = _to_inner(u0, Nx)
    lam = Q ** s
    uh = dst2(ui)
    dt = T / Nt
    b = _l1_b(alpha, Nt)
    b0 = b[0]
    dta = dt ** alpha
    hist = [uh]
    for n in range(1, Nt + 1):
        rhs = b[n - 1] * hist[0]
        for j in range(1, n):
            rhs = rhs + (b[n - j - 1] - b[n - j]) * hist[j]
        vh = rhs / (b0 + dta * lam)
        hist.append(vh)
    return _wrap(dst2(hist[-1]), Nx)


def solve_spacetime_variable(u0, alpha, s, nu, T, Nt, Nx, Q=None, return_inner=False):
    """Variable-coefficient explicit L1: nu(x,y) does not commute with (-Delta)^s; advance in physical space one time step at a time."""
    if Q is None:
        Q = spectral_eigvals(Nx)
    ui = _to_inner(u0, Nx)
    nu_in = nu[1:Nx, 1:Nx] if nu.shape[0] == Nx + 1 else nu
    Qs = Q ** s
    dt = T / Nt
    b = _l1_b(alpha, Nt)
    b0 = b[0]
    dta = dt ** alpha
    hist = [ui]
    for n in range(1, Nt + 1):
        rhs = b[n - 1] * hist[0]
        for j in range(1, n):
            rhs = rhs + (b[n - j - 1] - b[n - j]) * hist[j]
        fl = frac_lap_const(hist[-1], Qs)          # (-Delta)^s u_{n-1}
        u_new = (rhs - dta * nu_in * fl) / b0
        hist.append(u_new)
    res = hist[-1]
    return res if return_inner else _wrap(res, Nx)


def solve_spacetime_imex(u0, alpha, s, nu, T, Nt, Nx, Q=None, return_inner=False, n_inner=2):
    """Variable-coefficient (quasi-)fully implicit L1: split nu=c+(nu-c), with c the interior mean as the DST-diagonal implicit backbone; the variable-coefficient
    part receives n_inner Picard corrections toward full implicitness (unconditionally stable; Nt depends only on accuracy). n_inner=1 is IMEX."""
    if Q is None:
        Q = spectral_eigvals(Nx)
    ui = _to_inner(u0, Nx)
    nu_in = nu[1:Nx, 1:Nx] if nu.shape[0] == Nx + 1 else nu
    c = float(np.max(nu_in)); dc = nu_in - c
    Qs = Q ** s
    dt = T / Nt
    b = _l1_b(alpha, Nt)
    b0 = b[0]; dta = dt ** alpha
    denom = b0 + dta * c * Qs
    hist = [ui]
    for n in range(1, Nt + 1):
        base = b[n - 1] * hist[0]
        for j in range(1, n):
            base = base + (b[n - j - 1] - b[n - j]) * hist[j]
        x = hist[-1]
        for _ in range(n_inner):
            Lx = frac_lap_const(x, Qs)
            r = base - dta * dc * Lx
            x = dst2(dst2(r) / denom)
        hist.append(x)
    res = hist[-1]
    return res if return_inner else _wrap(res, Nx)
