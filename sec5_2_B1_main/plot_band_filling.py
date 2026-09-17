# -*- coding: utf-8 -*-
"""Band-filling demonstration figure: trained at 17^2 (K=10 modes), zero-shot to 129^2.
Shows radially averaged phase error: FrFNO fills out-of-band modes with the exact
analytic propagator, so its phase is more accurate than FNO's."""
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)

import os
ROOT = _PKG_ROOT
import sys; sys.path.insert(0, ROOT)
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import d_scan as DS
from d_scan import kl_basis, KMAX, PHI_CACHE
import frfno_core as FC
from frfno_core import DualNet, build_prop_table, prop_at, ub_full_batch, A_GRID, S_GRID, DEV, DT
import b1_burgers as B1
MEAN, SV = 0.0051, 0.2266
DA, DSNB = 0.10, 0.08

def build_nets():
    W = torch.load(os.path.join(ROOT, 'b1_weights.pt'), map_location=DEV)
    nets = {'FrFNO': DualNet(7, seed=1), 'FNO': DualNet(2, field_dim=2, seed=2)}
    for k in nets: nets[k].load_state_dict(W[k]); nets[k].eval()
    return nets

@torch.no_grad()
def pred(name, net, Nx, U0, NU, a, s, PT):
    out = []
    for i in range(len(a)):
        u0t = torch.tensor(U0[i:i+1], device=DEV, dtype=DT)
        nut = torch.tensor(NU[i:i+1], device=DEV, dtype=DT); aa, ss = float(a[i]), float(s[i])
        if name == 'FrFNO':
            gs = [prop_at(aa, ss, *PT)] + [prop_at(*z, *PT) for z in
                  [(aa-DA, ss), (aa+DA, ss), (aa, ss-DSNB), (aa, ss+DSNB)]]
            ubs = [ub_full_batch(U0[i:i+1], g, Nx) for g in gs]
            xb = torch.stack([u0t, ubs[0], nut, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            p = ubs[0].cpu().numpy() + SV*net(xb, (aa, ss)).cpu().numpy()
        elif name == 'FNO':
            xb = torch.stack([u0t, nut], -1)
            p = MEAN + SV*net(xb, (aa, ss)).cpu().numpy()
        out.append(p.reshape(Nx+1, Nx+1))
    return np.stack(out)

def radial_phase_error(fields_pred, fields_true, n=128):
    """Compute radially averaged absolute phase error between prediction and truth."""
    Up = fields_pred[:, 1:1+n, 1:1+n]
    Ut = fields_true[:, 1:1+n, 1:1+n]
    Fp = np.fft.fftn(Up, axes=(-2, -1))
    Ft = np.fft.fftn(Ut, axes=(-2, -1))
    kx = np.fft.fftfreq(n, d=1./n); KX, KY = np.meshgrid(kx, kx, indexing='ij')
    KR = np.sqrt(KX**2+KY**2).astype(int)
    phase_diff = np.angle(Fp * np.conj(Ft))  # shape: (Ntest, n, n)
    E_phase = np.array([np.abs(phase_diff[:, KR == k]).mean() for k in range(n//2)])
    return E_phase

def main():
    FC.T = B1.TN
    for Nx in (16, 128):
        Phi, e = kl_basis(Nx, KMAX); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Phi, e
    PT = {128: build_prop_table(128, Nt=400)}
    nets = build_nets()
    rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=B1.NTEST)
    a = rng.uniform(A_GRID.min(), A_GRID.max(), B1.NTEST).astype(np.float32)
    s = rng.uniform(S_GRID.min(), S_GRID.max(), B1.NTEST).astype(np.float32)
    xi = rng.randn(B1.NTEST, KMAX).astype(np.float32)
    Phi, e = PHI_CACHE[(128, 'Phi')], PHI_CACHE[(128, 'e')]
    Ftr, U0, NU = B1.true_field_nl(128, 1600, seeds, a, s, xi, Phi, e)
    true = Ftr.reshape(-1, 129, 129)
    P = {m: pred(m, nets[m], 128, U0, NU, a, s, PT[128]) for m in nets}
    E_phase_fr = radial_phase_error(P['FrFNO'], true)
    E_phase_fno = radial_phase_error(P['FNO'], true)
    k = np.arange(64)
    K_train = 10

    # Print actual values for inspection
    print('\nRadial phase error at selected k (radians):')
    print(f'{"k":>4} {"FrFNO":>10} {"FNO":>10}')
    for kk in [5, 8, 10, 12, 15, 20, 30, 40]:
        print(f'{kk:>4d} {E_phase_fr[kk]:>10.3f} {E_phase_fno[kk]:>10.3f}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.plot(k[1:], E_phase_fr[1:], color='#d62728', linestyle='-', linewidth=2.5, label='FrFNO')
    ax1.plot(k[1:], E_phase_fno[1:], color='#1f77b4', linestyle=':', linewidth=2.5, label='FNO')
    ax1.axvline(K_train, color='gray', ls='--', lw=1.5)
    ax1.text(K_train+0.5, 0.1, f'K={K_train}\n(training band)', fontsize=9, color='gray')
    ax1.axvspan(K_train, 64, color='red', alpha=.06)
    ax1.text(30, 0.1, 'out-of-band\n(k>K)', fontsize=10, color='red', ha='center')
    ax1.set_xlabel('wavenumber |k|'); ax1.set_ylabel('radially averaged phase error (rad)')
    ax1.set_title('Per-mode phase error: 17$^2$ train $\\rightarrow$ 129$^2$ zero-shot')
    ax1.legend(fontsize=11); ax1.grid(True, alpha=.3)

    mask = (k >= K_train) & (k < 45)
    ax2.plot(k[mask], E_phase_fr[mask], color='#d62728', linestyle='-', linewidth=2.5, marker='o', markersize=6, label='FrFNO')
    ax2.plot(k[mask], E_phase_fno[mask], color='#1f77b4', linestyle=':', linewidth=2.5, marker='s', markersize=6, label='FNO')
    ax2.set_xlabel('wavenumber |k|'); ax2.set_ylabel('radially averaged phase error (rad)')
    ax2.set_title('Out-of-band zoom: FrFNO phase is more accurate')
    ax2.legend(fontsize=11); ax2.grid(True, alpha=.3)

    plt.tight_layout()
    out = r'D:\科研\paper_JCP2\figs\fig_band_filling.png'
    plt.savefig(out, dpi=200); print('saved', out)

if __name__ == '__main__':
    main()
