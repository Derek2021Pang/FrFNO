# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""B3 radial energy-spectrum mechanism figure (B1 nonlinear, trained at 17^2 -> zero-shot 129^2):
compare the radial mean power spectra of the truth and each model prediction; the vertical line marks the resolvable wavenumber cap at training resolution, with the extrapolation zone to its right.
Also quantify the log-spectrum error over the extrapolation zone (k=16..48)."""
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
from models_surrogate import CNOWrap
from models_extra import PDNO2d
import b1_burgers as B1
MEAN, SV = 0.0051, 0.2266
DA, DSNB = 0.10, 0.08

def build_nets():
    W = torch.load(os.path.join(ROOT, 'b1_weights.pt'), map_location=DEV)
    nets = {'FrFNO': DualNet(7, seed=1), 'FNO': DualNet(2, field_dim=2, seed=2),
            'PINO': DualNet(2, field_dim=2, seed=7), 'PDNO': PDNO2d(seed=6),
            'CNO': CNOWrap(17, ch=32, use_bn=False, seed=3).to(DEV).float()}
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
        elif name in ('FNO', 'PINO', 'PDNO'):
            xb = torch.stack([u0t, nut], -1)
            p = MEAN + SV*net(xb, (aa, ss)).cpu().numpy()
        else:
            x = torch.stack([u0t, nut], 1); cond = torch.tensor([[aa, ss]], device=DEV, dtype=DT)
            p = MEAN + SV*net(x, cond).cpu().numpy()
        out.append(p.reshape(Nx+1, Nx+1))
    return np.stack(out)

def radial_spectrum(fields, n=128):
    U = fields[:, 1:1+n, 1:1+n]
    Ff = np.fft.fftn(U, axes=(-2, -1)); P = np.abs(Ff)**2
    kx = np.fft.fftfreq(n, d=1./n); KX, KY = np.meshgrid(kx, kx, indexing='ij')
    KR = np.sqrt(KX**2+KY**2).astype(int)
    E = np.array([P[:, KR == k].mean() for k in range(n//2)])
    return E

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
    P['Truth'] = true
    E = {m: radial_spectrum(P[m]) for m in ['Truth', 'FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO']}
    k = np.arange(64)
    # Piecewise quantification: (1) near-extrapolation energy-dominated zone k9..18, energy-weighted log error;
    #          (2) far tail k20..45, equal-weight shape error (geometric-mean ratio; catches spurious high frequencies / over-smoothing) and whether the tail decays.
    def wlog(lo, hi):
        w = E['Truth'][lo:hi+1]
        return {m: np.sum(w*np.abs(np.log(E[m][lo:hi+1]+1e-30)-np.log(E['Truth'][lo:hi+1]+1e-30)))/np.sum(w)
                for m in ['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO']}
    def tail_shape(lo, hi):
        out = {}
        t = E['Truth'][lo:hi+1]
        for m in ['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO', 'Truth']:
            seg = E[m][lo:hi+1]
            out[m] = (0.0 if m == 'Truth' else np.mean(np.abs(np.log(seg+1e-30)-np.log(t+1e-30))),
                      np.polyfit(np.arange(lo, hi+1), np.log(seg+1e-30), 1)[0])
        return out
    wn = wlog(9, 18); ts = tail_shape(20, 45)
    print('Near-extrapolation k9..18 energy-weighted spectrum error (smaller is better):')
    for m in ['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO']:
        print('  %-6s %.3f' % (m, wn[m]))
    print('Far-tail k20..45 shape error / log slope (truth %.3f, closer is better; a positive slope means a spurious high-freq upturn):' % ts['Truth'][1])
    for m in ['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO']:
        print('  %-6s shape=%.3f  slope=%.3f' % (m, ts[m][0], ts[m][1]))
    colors = {'Truth': 'k', 'FrFNO': '#d62728', 'FNO': '#1f77b4', 'PINO': '#2ca02c',
              'PDNO': '#9467bd', 'CNO': '#ff7f0e'}
    plt.figure(figsize=(7.2, 5.4))
    plt.axvspan(8, 18, color='green', alpha=.06); plt.axvspan(20, 45, color='red', alpha=.06)
    for m in ['Truth', 'FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO']:
        lw = 2.4 if m in ('Truth', 'FrFNO') else 1.6
        ls = '-' if m == 'Truth' else '--'
        plt.loglog(k[1:], E[m][1:], ls, color=colors[m], lw=lw, label=m)
    plt.axvline(8, color='gray', ls=':', lw=1.5)
    plt.text(8.4, plt.ylim()[1]*0.5, 'train resol.\n|k|=8', fontsize=9, color='gray')
    plt.xlim(1, 64)
    plt.xlabel('wavenumber |k|'); plt.ylabel('radial power spectrum E(k)')
    plt.title('Nonlinear fractional Burgers: 17$^2$ train $\\rightarrow$ 129$^2$ zero-shot')
    plt.legend(); plt.grid(True, which='both', alpha=.3); plt.tight_layout()
    out = os.path.join(ROOT, '..', 'paper_JCP', 'figs', 'fig_radial_spectrum.png')
    plt.savefig(out, dpi=200); print('saved', out)

if __name__ == '__main__':
    main()
