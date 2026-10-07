"""
Fractional PDE study: common utilities (rebuilt from the archived stage-by-stage mathematical description)
- generate_multiscale_initial: multiscale random sinusoidal initial data on a homogeneous Dirichlet domain
  K x K random modes; high-frequency coefficients decay as (i^2+j^2)^(-1/2); the boundary is naturally zero.
Grid convention: Nx intervals -> (Nx+1)x(Nx+1) grid points; interior points are [1:Nx,1:Nx] (Nx-1 per dimension).
"""
import numpy as np


def generate_multiscale_initial(Nx, K=8, seed=0, target_rms=0.5, rng=None):
    """Generate a (Nx+1,Nx+1) multiscale random initial field with zero boundaries.

    u0(x,y) = sum_{i,j=1..K} c_ij / sqrt(i^2+j^2) * sin(i*pi*x) sin(j*pi*y)
    Finally rescale globally to target_rms (the RMS over interior points) to keep scale consistent across samples/resolutions.
    """
    if rng is None:
        rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 1.0, Nx + 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    u = np.zeros((Nx + 1, Nx + 1), dtype=np.float64)
    for i in range(1, K + 1):
        sx = np.sin(i * np.pi * X)
        for j in range(1, K + 1):
            c = rng.normal()
            u += c / np.sqrt(i ** 2 + j ** 2) * sx * np.sin(j * np.pi * Y)
    inner = u[1:Nx, 1:Nx]
    rms = np.sqrt(np.mean(inner ** 2)) + 1e-12
    u *= target_rms / rms
    # Enforce strict zero boundaries (sines are theoretically zero; pin them once more numerically)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    return u
