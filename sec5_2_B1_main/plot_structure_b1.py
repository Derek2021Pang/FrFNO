# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""B3 supplement: mechanistic evidence at the structure/phase level (power spectra discard phase; the L2 advantage should arise from spatial structure and phase).
Top row: representative 129^2 snapshots (truth + 5 models, shared color scale);
bottom-left: x-profile along the y-midline (checks front-position alignment); bottom-right: radial phase error vs wavenumber.
Quantification: mean phase error (k1..32) and normalized cross-correlation (NCC)."""
import os
ROOT = _PKG_ROOT
import sys; sys.path.insert(0, ROOT)
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import d_scan as DS
from d_scan import kl_basis, KMAX, PHI_CACHE
import frfno_core as FC
from frfno_core import build_prop_table, A_GRID, S_GRID
import b1_burgers as B1
from plot_spectrum_b1 import build_nets, pred

def phase_err(fm, ft, n=128):
    Um = np.fft.fftn(fm[:, 1:1+n, 1:1+n], axes=(-2, -1))
    Ut = np.fft.fftn(ft[:, 1:1+n, 1:1+n], axes=(-2, -1))
    dphi = np.angle(Um*np.conj(Ut))
    kx = np.fft.fftfreq(n, d=1./n); KX, KY = np.meshgrid(kx, kx, indexing='ij')
    KR = np.sqrt(KX**2+KY**2).astype(int)
    return np.array([np.abs(dphi[:, KR == k]).mean() for k in range(n//2)])

def ncc(fm, ft):
    out = []
    for a, b in zip(fm, ft):
        a = a-a.mean(); b = b-b.mean()
        out.append((a*b).sum()/np.sqrt((a**2).sum()*(b**2).sum()+1e-30))
    return np.mean(out)

def main():
    FC.T = B1.TN
    for Nx in (16, 128):
        Phi, e = kl_basis(Nx, KMAX); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Phi, e
    PT = {128: build_prop_table(128, Nt=400)}; nets = build_nets()
    rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=B1.NTEST)
    a = rng.uniform(A_GRID.min(), A_GRID.max(), B1.NTEST).astype(np.float32)
    s = rng.uniform(S_GRID.min(), S_GRID.max(), B1.NTEST).astype(np.float32)
    xi = rng.randn(B1.NTEST, KMAX).astype(np.float32)
    Phi, e = PHI_CACHE[(128, 'Phi')], PHI_CACHE[(128, 'e')]
    Ftr, U0, NU = B1.true_field_nl(128, 1600, seeds, a, s, xi, Phi, e)
    true = Ftr.reshape(-1, 129, 129)
    Ms = ['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO']
    P = {m: pred(m, nets[m], 128, U0, NU, a, s, PT[128]) for m in Ms}
    # Pick a representative sample: the L2 advantage of FrFNO over PINO is close to the median
    l2 = {m: np.linalg.norm(P[m]-true, axis=(1, 2))/(np.linalg.norm(true, axis=(1, 2))+1e-30) for m in Ms}
    gap = l2['PINO']-l2['FrFNO']; idx = int(np.argsort(gap)[len(gap)//2])
    print('representative sample idx=%d (a=%.3f s=%.3f); per-sample averages:' % (idx, a[idx], s[idx]))
    PE = {m: phase_err(P[m], true) for m in Ms}
    for m in Ms:
        print('  %-6s L2=%.3f%%  phase error (k1-32)=%.3f rad  NCC=%.4f'
              % (m, 100*l2[m].mean(), PE[m][1:33].mean(), ncc(P[m], true)))
    colors = {'FrFNO': '#d62728', 'FNO': '#1f77b4', 'PINO': '#2ca02c', 'PDNO': '#9467bd', 'CNO': '#ff7f0e'}
    fig = plt.figure(figsize=(14, 8.2))
    panels = [('Truth', true)] + [(m, P[m]) for m in Ms]
    vmin, vmax = np.percentile(true[idx], [4, 96])
    for j, (nm, F) in enumerate(panels):
        ax = fig.add_subplot(2, 6, j+1); im = ax.imshow(F[idx], vmin=vmin, vmax=vmax, cmap='RdBu_r')
        ax.set_title(nm, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    axp = fig.add_subplot(2, 2, 3)
    jm = 64; xx = np.linspace(0, 1, 129)
    axp.plot(xx, true[idx, :, jm], 'k', lw=2.4, label='Truth')
    for m in Ms: axp.plot(xx, P[m][idx, :, jm], '--', color=colors[m], lw=1.5, label=m)
    axp.set_title('x-profile at mid-y'); axp.set_xlabel('x'); axp.legend(fontsize=8); axp.grid(alpha=.3)
    axx = fig.add_subplot(2, 2, 4); kk = np.arange(1, 40)
    for m in Ms: axx.plot(kk, PE[m][1:40], '--', color=colors[m], lw=1.6, label=m)
    axx.axvline(8, color='gray', ls=':'); axx.set_title('mean phase error vs |k|')
    axx.set_xlabel('wavenumber |k|'); axx.set_ylabel('|phase diff| (rad)'); axx.legend(fontsize=8); axx.grid(alpha=.3)
    fig.tight_layout(); out = os.path.join(ROOT, 'b1_structure.png')
    plt.savefig(out, dpi=190); print('saved', out)

if __name__ == '__main__':
    main()
