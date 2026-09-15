"""
Variable diffusion-coefficient field generation (rebuilt from the archived stages)
- smooth_nu_field: a superposition of low-frequency sines -> a positive log-normal field
  The caller usually then applies nu = nu / nu.mean() to normalize the mean diffusivity to 1,
  the analytic baseline uniformly uses nu0=1 and the network learns only the spatial-heterogeneity residual.
"""
import numpy as np


def smooth_nu_field(Nx, seed=0, magnitude=0.4, n_modes=4, rng=None):
    """Generate a (Nx+1,Nx+1) smooth, positive, log-normal diffusivity field.

    g(x,y) is a superposition of n_modes low-frequency terms sin(kx*pi*x + ky*pi*y + phase), standardized to
    unit standard deviation; nu = exp(magnitude * g) is everywhere positive and spatially smooth.
    """
    if rng is None:
        rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 1.0, Nx + 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    g = np.zeros((Nx + 1, Nx + 1), dtype=np.float64)
    for _ in range(n_modes):
        kx = rng.randint(1, 4)
        ky = rng.randint(1, 4)
        amp = rng.uniform(0.5, 1.0)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        g += amp * np.sin(kx * np.pi * X + ky * np.pi * Y + ph)
    g -= g.mean()
    g /= g.std() + 1e-12
    nu = np.exp(magnitude * g)
    return nu
