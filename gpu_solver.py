"""
GPU-batched space-time two-orders variable-coefficient diffusion solver (torch).
Equation: ^C D_t^a u = -nu(x,y) (-Delta)^s u, homogeneous Dirichlet.
- Advances B trajectories (u0,nu,a,s) in one parallel pass (a,s may differ per sample) for high-resolution reference solutions.
- DST-I (ortho) is built with torch.fft (cross-checked against scipy.fft.dst(type=1,norm='ortho'));
  the orthogonal DST-I is self-inverse, so the same function serves both forward and inverse.
- The L1 history term uses a preallocated history tensor plus one batched einsum weighting, avoiding O(Nt^2) Python loops.
Memory: Nt*B*(Nx-1)^2*4 bytes; the caller batches by resolution.
"""
import os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))


def torch_dst1(x, dim=-1):
    """1-D DST-I (ortho); x is a torch tensor with arbitrary batch dims, transformed along dim."""
    N = x.shape[dim]
    # Odd-symmetric extension z=[0, x, 0, -flip(x)], length 2(N+1); DST-I = -Im of the FFT on segments 1..N
    zeros_shape = list(x.shape); zeros_shape[dim] = 1
    z = torch.cat([
        torch.zeros(zeros_shape, dtype=x.dtype, device=x.device),
        x,
        torch.zeros(zeros_shape, dtype=x.dtype, device=x.device),
        -torch.flip(x, dims=[dim]),
    ], dim=dim)
    Z = torch.fft.fft(z, dim=dim)
    idx = torch.arange(1, N + 1, device=x.device)
    Y = -torch.index_select(Z.imag, dim, idx)          # = 2*sum x sin (unnormalized DST-I)
    # Orthogonal DST-I entries are sqrt(2/(N+1))*sin, so relative to the 2*sum form the scaling factor is 1/sqrt(2(N+1))
    return Y / np.sqrt(2.0 * (N + 1))


def dst2_torch(u):
    """Double DST-I of a 2-D interior field (B,n,n); self-inverse, hence it is also the inverse transform."""
    return torch_dst1(torch_dst1(u, dim=-1), dim=-2)


def spectral_eigvals_torch(Nx, device, dtype=torch.float32):
    h = 1.0 / Nx
    m = torch.arange(1, Nx, device=device, dtype=dtype)
    s = (4.0 / h ** 2) * torch.sin(np.pi * h * m / 2.0) ** 2
    QX, QY = torch.meshgrid(s, s, indexing='ij')
    return QX + QY


def l1_b_torch(alpha, Nt, device, dtype=torch.float32):
    """Standard L1 weights b_j=[(j+1)^(1-alpha)-j^(1-alpha)]/Gamma(2-alpha)."""
    j = torch.arange(Nt, device=device, dtype=dtype)
    bp1 = (j + 1.0) ** (1.0 - alpha)
    bj = torch.where(j == 0, torch.zeros_like(j), j ** (1.0 - alpha))
    return (bp1 - bj) / torch.exp(torch.lgamma(torch.tensor(2.0 - alpha, device=device, dtype=dtype)))


def gpu_solve_batch(u0, nu, alpha, s, T, Nt, Nx, Q=None, device='cuda',
                    dtype=torch.float32, verbose=False):
    """Advance B sample trajectories to T in one batch.
    u0,nu: (B,Nx+1,Nx+1) or (B,Nx-1,Nx-1); alpha,s: (B,). Returns (B,Nx+1,Nx+1).
    """
    if not torch.is_tensor(u0): u0 = torch.as_tensor(u0)
    if not torch.is_tensor(nu): nu = torch.as_tensor(nu)
    u0 = u0.to(device=device, dtype=dtype); nu = nu.to(device=device, dtype=dtype)
    alpha = torch.as_tensor(alpha, device=device, dtype=dtype).reshape(-1)
    s = torch.as_tensor(s, device=device, dtype=dtype).reshape(-1)
    B = u0.shape[0]
    if u0.shape[-1] == Nx + 1:
        ui = u0[:, 1:Nx, 1:Nx].contiguous(); nu_in = nu[:, 1:Nx, 1:Nx].contiguous()
    else:
        ui = u0.contiguous(); nu_in = nu.contiguous()
    n = Nx - 1
    if Q is None: Q = spectral_eigvals_torch(Nx, device, dtype)
    # Per-sample Q^s (s differs across samples)
    Qs = Q[None] ** s[:, None, None]                       # (B,n,n)
    # L1 coefficients, batched and vectorized (no per-sample Python loop)
    jj = torch.arange(Nt, device=device, dtype=dtype)[None]
    ac = alpha[:, None]
    bp1 = (jj + 1.0) ** (1.0 - ac)
    bj = torch.where(jj == 0, torch.zeros_like(jj), jj ** (1.0 - ac))
    b = (bp1 - bj) / torch.exp(torch.lgamma(2.0 - alpha))[:, None]   # (B,Nt) standard L1
    dt = T / Nt
    dta = (dt ** alpha)[:, None, None]                     # (B,1,1)
    hist = torch.empty((Nt, B, n, n), device=device, dtype=dtype)
    hist[0] = ui
    for k in range(1, Nt + 1):
        rhs = b[:, k - 1, None, None] * hist[0]
        if k > 1:
            # w_{b,j} = b[b,k-j-1]-b[b,k-j], j=1..k-1  -> aligns with hist[1:k]
            jj = torch.arange(1, k, device=device)
            w = b[:, k - jj - 1] - b[:, k - jj]            # (B,k-1)
            rhs = rhs + torch.einsum('bj,jbxy->bxy', w, hist[1:k])
        fl = dst2_torch(Qs * dst2_torch(hist[k - 1]))      # (-Δ)^s u_{k-1}
        un = (rhs - dta * nu_in * fl) / b[:, 0][:, None, None]  # standard L1: b0=1/Gamma
        if k < Nt: hist[k] = un
    out = torch.zeros((B, Nx + 1, Nx + 1), device=device, dtype=dtype)
    out[:, 1:Nx, 1:Nx] = un
    return out


def gpu_solve_imex_batch(u0, nu, alpha, s, T, Nt, Nx, Q=None,
                         device='cuda', dtype=torch.float32, n_inner=2):
    """
    Batched (quasi-)fully implicit GPU solution of variable-coefficient space-time fractional diffusion ^CD_t^alpha u = -nu(x)(-Delta)^s u.
    Split nu = c + (nu-c), with c the interior mean as the implicit backbone (diagonal DST inversion); the variable-coefficient part uses n_inner
    Picard fixed-point corrections approach full implicitness -> unconditionally stable for dissipative equations; Nt is set by temporal accuracy rather than grid stability.
    n_inner=1 reduces to IMEX; default 2. Inputs/outputs are identical to gpu_solve_batch.
    """
    B = len(alpha); n = Nx - 1
    if Q is None:
        Q = spectral_eigvals_torch(Nx, device, dtype)
    dt = T / Nt
    alpha = torch.as_tensor(np.asarray(alpha), device=device, dtype=dtype)
    s = torch.as_tensor(np.asarray(s), device=device, dtype=dtype)
    dta = (dt ** alpha)[:, None, None]                      # (B,1,1)
    jj = torch.arange(Nt, device=device, dtype=dtype)[None]
    ac = alpha[:, None]
    bp1 = (jj + 1.0) ** (1.0 - ac)
    bj = torch.where(jj == 0, torch.zeros_like(jj), jj ** (1.0 - ac))
    b = (bp1 - bj) / torch.exp(torch.lgamma(2.0 - alpha))[:, None]     # (B,Nt) standard L1
    Qs = Q[None] ** s[:, None, None]                        # (B,n,n)
    ui = torch.as_tensor(u0[:, 1:Nx, 1:Nx], device=device, dtype=dtype)
    ni = torch.as_tensor(nu[:, 1:Nx, 1:Nx], device=device, dtype=dtype)
    c = ni.amax(dim=(1, 2), keepdim=True)               # The implicit backbone takes the upper bound of nu to guarantee Picard contraction
    hist = torch.zeros((Nt + 1, B, n, n), device=device, dtype=dtype)
    hist[0] = ui
    b0 = b[:, 0][:, None, None]
    denom = b0 + dta * c * Qs                               # Diagonal spectral-domain backbone operator
    for k in range(1, Nt + 1):
        base = b[:, k - 1][:, None, None] * ui
        if k > 1:
            j = torch.arange(1, k, device=device)
            ww = b[:, k - j - 1] - b[:, k - j]
            base = base + torch.einsum('bj,jbxy->bxy', ww, hist[1:k])
        x = hist[k - 1]                                     # Picard initial guess = the previous time level
        for _ in range(n_inner):
            Lx = dst2_torch(Qs * dst2_torch(x))
            r = base - dta * (ni - c) * Lx
            x = dst2_torch(dst2_torch(r) / denom)
        hist[k] = x
    out = torch.zeros((B, Nx + 1, Nx + 1), device=device, dtype=dtype)
    out[:, 1:Nx, 1:Nx] = hist[Nt]
    return out


if __name__ == '__main__':
    # Cross-check the batched GPU solution against the CPU numpy solution
    import sys, time
    sys.path.insert(0, os.path.join(ROOT, 'fractional_pde_study'))
    from solver import generate_multiscale_initial
    from solver_variable import smooth_nu_field
    from solver_spacetime_fractional import solve_spacetime_variable
    from scipy.fft import dst as spdst

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device =', dev, torch.cuda.get_device_name(0) if dev == 'cuda' else '')

    # --- DST cross-check ---
    rng = np.random.RandomState(0)
    xn = rng.randn(6, 5, 7)
    xt = torch.tensor(xn, dtype=torch.float64)
    g = torch_dst1(torch_dst1(xt, dim=-1), dim=-2).numpy()
    c = spdst(spdst(xn, type=1, axis=-1, norm='ortho'), type=1, axis=-2, norm='ortho')
    print('max abs error of double DST-I vs scipy =', np.abs(g - c).max())

    for Nx, Nt in [(16, 400), (64, 1600)]:
        T = 0.01
        a_list = [0.6, 0.75, 0.9]; s_list = [0.4, 0.6, 0.75]
        B = len(a_list)
        U0 = np.stack([generate_multiscale_initial(Nx, K=8, seed=100 + i) for i in range(B)])
        NU = np.stack([smooth_nu_field(Nx, seed=200 + i, magnitude=.4, n_modes=4) for i in range(B)])
        NU = NU / NU.mean(axis=(1, 2), keepdims=True)
        # CPU, sample by sample
        t0 = time.time(); cpu = []
        for i in range(B):
            cpu.append(solve_spacetime_variable(U0[i], a_list[i], s_list[i], NU[i], T, Nt, Nx))
        cpu = np.stack(cpu); tcpu = time.time() - t0
        # GPU, batched
        torch.cuda.synchronize() if dev == 'cuda' else None
        t0 = time.time()
        gpu = gpu_solve_batch(U0, NU, a_list, s_list, T, Nt, Nx, device=dev,
                              dtype=torch.float64).cpu().numpy()
        torch.cuda.synchronize() if dev == 'cuda' else None
        tgpu = time.time() - t0
        err = np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu)
        print(f'Nx={Nx},Nt={Nt},B={B}: GPU-vs-CPU relL2={err:.2e} | CPU {tcpu:.1f}s GPU {tgpu:.1f}s speedup {tcpu/max(tgpu,1e-6):.1f}x')
