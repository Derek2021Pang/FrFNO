# -*- coding: utf-8 -*-
"""Validate Exp2 spectral tail slope.
Compare:
- true residual r_true = u_true - u_base vs network residual r_net = u_pred - u_base
- radial definition (kr >= K/N) vs square definition (|kx|>=K/N or |ky|>=K/N)
- slope grouped by s (low/mid/high) to check slope ~ -2s
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SEC = _os.path.dirname(_HERE); _PKG_ROOT = _os.path.dirname(_SEC)
for _p in (_HERE, _SEC, _PKG_ROOT):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import os, sys
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'theory_closure'))
import numpy as np, torch
import frfno_legacy as LG
from frfno_legacy import DualNet, DEV, DT, prop_at, ub_full_batch
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
import surrogate_compare as SC
from theory_verify import build_pt_eig
import p0_platform as P

OUT = P.OUT
REF_NPZ = P.REF_NPZ
L = []
def log(s): print(s, flush=True); L.append(str(s)); open(os.path.join(OUT, 'exp5_validate.log'), 'w', encoding='utf-8').write('\n'.join(L))

def spectral_tail(rint, K, Nx, mode='radial'):
    """Compute sigma_K(r) = ||(I-P_K)r|| / ||r||.
    rint: (N, N) interior field
    K: spectral modes
    mode: 'radial' = kr >= K/N, 'square' = |kx|>=K/N or |ky|>=K/N
    """
    Fk = np.fft.fft2(rint)
    power = np.abs(Fk)**2
    ny, nx = rint.shape
    ky, kx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing='ij')
    K_cycles = K / ny
    if mode == 'radial':
        kr = np.sqrt(kx**2 + ky**2)
        mask_tail = kr >= K_cycles
    else:  # square
        mask_tail = (np.abs(kx) >= K_cycles) | (np.abs(ky) >= K_cycles)
    total = power.sum()
    tail = power[mask_tail].sum()
    return tail / (total + 1e-30)

@torch.no_grad()
def main():
    zr = dict(np.load(REF_NPZ, allow_pickle=True))
    F513, seeds, a, s, xi = zr['F'], zr['seeds'], zr['a'], zr['s'], zr['xi']
    Nx = 64
    Phi, e = P.get_phi(Nx)
    U0 = np.stack([LG.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
    NU = P.make_nu(Nx, xi, Phi, e)
    Fds = P.downsample(F513, Nx)  # (48, (N+1)^2)
    Fds2d = Fds.reshape(len(a), Nx+1, Nx+1)
    PT = build_pt_eig(Nx, 3200, P.AD, P.SDG)

    # Load one FrFNO model (K=8) for network residual
    ck = torch.load(os.path.join(OUT, 'kscan_N64_K8.pt'), map_location=DEV)
    MEAN = float(ck.get('MEAN', 0.0060)); SV = float(ck.get('SV', 0.2620))
    SC.MEAN, SC.SV = MEAN, SV
    net = DualNet(7, modes=8, pad=4, field_dim=1, seed=1).to(DEV).float()
    net.load_state_dict(ck['FrFNO']); net.eval()

    Ks = [4, 6, 8, 10, 12, 16, 20]
    results = {'true_radial': {K: [] for K in Ks}, 'true_square': {K: [] for K in Ks},
               'net_radial': {K: [] for K in Ks}, 'net_square': {K: [] for K in Ks}}
    svals_by_s = {'low': [], 'mid': [], 'high': []}

    for i in range(len(a)):
        aa, ss = float(a[i]), float(s[i])
        # group by s
        if ss < 0.45: svals_by_s['low'].append(i)
        elif ss < 0.65: svals_by_s['mid'].append(i)
        else: svals_by_s['high'].append(i)

        # true residual: u_true - u_base
        gs = [prop_at(aa, ss, *PT)]
        ubs = [ub_full_batch(U0[i:i+1], g, Nx) for g in gs]
        u_base = ubs[0][0].cpu().numpy()  # (N+1, N+1)
        u_true = Fds2d[i]   # (N+1, N+1)
        r_true = (u_true - u_base)[1:, 1:]  # interior (N, N)

        # network residual: u_pred - u_base
        u0t = torch.tensor(U0[i:i+1], device=DEV, dtype=DT)
        nut = torch.tensor(NU[i:i+1], device=DEV, dtype=DT)
        gs_all = [prop_at(aa, ss, *PT)]+[prop_at(*z, *PT) for z in [(aa-0.10, ss), (aa+0.10, ss), (aa, ss-0.08), (aa, ss+0.08)]]
        ubs_all = [ub_full_batch(U0[i:i+1], g, Nx) for g in gs_all]
        xb = torch.stack([u0t, ubs_all[0], nut, ubs_all[1], ubs_all[2], ubs_all[3], ubs_all[4]], -1)
        r_net = (SV * net(xb, (aa, ss))).cpu().numpy()[0][1:, 1:]

        for K in Ks:
            results['true_radial'][K].append(spectral_tail(r_true, K, Nx, 'radial'))
            results['true_square'][K].append(spectral_tail(r_true, K, Nx, 'square'))
            results['net_radial'][K].append(spectral_tail(r_net, K, Nx, 'radial'))
            results['net_square'][K].append(spectral_tail(r_net, K, Nx, 'square'))

    # Summary: mean sigma per K, and log-log slope
    log('='*80)
    log('Spectral tail validation (N0=65^2, 48 test samples)')
    log('='*80)
    for mode in ['true_radial', 'true_square', 'net_radial', 'net_square']:
        means = [np.mean(results[mode][K]) for K in Ks]
        valid = [(K, m) for K, m in zip(Ks, means) if m > 1e-8]
        if len(valid) >= 3:
            Karr = np.array([v[0] for v in valid], dtype=float)
            marr = np.array([v[1] for v in valid])
            slope, intercept = np.polyfit(np.log10(Karr), np.log10(marr), 1)
        else:
            slope = float('nan')
        log(f'\n--- {mode} (slope={slope:.3f}) ---')
        log(f'{"K":>4} ' + ' '.join(f'{K:>8}' for K in Ks))
        log(f'{"sigma":>4} ' + ' '.join(f'{m:8.2e}' for m in means))

    # Slope grouped by s (true residual, radial)
    log('\n' + '='*80)
    log('Slope grouped by s (true residual, radial definition)')
    log('='*80)
    for grp, idxs in svals_by_s.items():
        if len(idxs) < 3: continue
        s_mean = np.mean([s[i] for i in idxs])
        means = []
        for K in Ks:
            vals = [results['true_radial'][K][i] for i in idxs]
            means.append(np.mean(vals))
        valid = [(K, m) for K, m in zip(Ks, means) if m > 1e-8]
        if len(valid) >= 3:
            Karr = np.array([v[0] for v in valid], dtype=float)
            marr = np.array([v[1] for v in valid])
            slope, _ = np.polyfit(np.log10(Karr), np.log10(marr), 1)
        else:
            slope = float('nan')
        log(f'  {grp} s (n={len(idxs)}, s_mean={s_mean:.3f}): slope={slope:.3f}, theory -2s={-2*s_mean:.3f}')

    log('\nVALIDATION DONE')

if __name__ == '__main__':
    main()
