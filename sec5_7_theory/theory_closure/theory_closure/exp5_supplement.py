# -*- coding: utf-8 -*-
"""Exp5 supplement: three experiments to verify theory.
Exp1: N->infinity convergence curve (baseline K=10,N0=17^2 at N=17/33/65/129/257)
Exp2: residual spectral tail sigma_K(r) ~ K^{-2s} slope (N0=65^2, K=4/6/8/10/12 FrFNO)
Exp3: K-as-bottleneck floor (N0=65^2, K=4/6/8/10/12, FrFNO+FNO at N=257)

Usage: python exp5_supplement.py [eval|spec|plot|all]
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SEC = _os.path.dirname(_HERE); _PKG_ROOT = _os.path.dirname(_SEC)
for _p in (_HERE, _SEC, _PKG_ROOT):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import os, sys, time
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'theory_closure'))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import frfno_legacy as LG
from frfno_legacy import DualNet, DEV, DT, prop_at, ub_full_batch, A_GRID, S_GRID
from gpu_solver import spectral_eigvals_torch
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
import surrogate_compare as SC
from theory_verify import build_pt_eig
import p0_platform as P

OUT = P.OUT
REF_NPZ = P.REF_NPZ
EVAL_NPZ = os.path.join(OUT, 'exp5_supp_eval.npz')
SPEC_NPZ = os.path.join(OUT, 'exp5_supp_spec.npz')
L = []
def log(s): print(s, flush=True); L.append(str(s)); open(os.path.join(OUT, 'exp5_supp.log'), 'w', encoding='utf-8').write('\n'.join(L))
def nz(p): return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else {}

# N0=65^2 K-scan configs (newly trained)
N64_CONFIGS = [(4, 'kscan_N64_K4.pt'), (6, 'kscan_N64_K6.pt'),
               (8, 'kscan_N64_K8.pt'), (10, 'trNx64_K10.pt'),
               (12, 'kscan_N64_K12.pt')]
# Evaluation tiers: add N=33 (Nx=32) for Exp1 convergence curve
RES_FULL = [(16, 3200), (32, 3200), (64, 3200), (128, 3200), (256, 3200)]

def load_nets(wpath, K, pad=4):
    ck = torch.load(wpath, map_location=DEV)
    MEAN = float(ck.get('MEAN', 0.0060)); SV = float(ck.get('SV', 0.2620))
    p = int(ck.get('pad', pad))
    nets = {}
    nets['FrFNO'] = DualNet(7, modes=K, pad=p, field_dim=1, seed=1).to(DEV).float()
    nets['FrFNO'].load_state_dict(ck['FrFNO']); nets['FrFNO'].eval()
    nets['FNO'] = DualNet(2, modes=K, pad=p, field_dim=0, seed=2).to(DEV).float()
    nets['FNO'].load_state_dict(ck['FNO']); nets['FNO'].eval()
    return nets, MEAN, SV, p

@torch.no_grad()
def stage_eval():
    zr = nz(REF_NPZ)
    if not zr or zr.get('n_done', 0) < P.NTE: log('ref not ready'); return
    F513, seeds, a, s, xi = zr['F'], zr['seeds'], zr['a'], zr['s'], zr['xi']
    ze = nz(EVAL_NPZ); E = ze['E'].item() if ze and 'E' in ze else {}

    # Evaluate N0=65^2 K-scan models at all tiers
    for K, wname in N64_CONFIGS:
        wpath = os.path.join(OUT, wname)
        if not os.path.exists(wpath): log(f'MISSING {wname}, skip'); continue
        nets, MEAN, SV, pad = load_nets(wpath, K)
        SC.MEAN, SC.SV = MEAN, SV
        log(f'Loaded N64 K={K}: MEAN={MEAN:.4f} SV={SV:.4f} pad={pad}')
        for Nx, Nt in RES_FULL:
            # Skip if K exceeds spectral capacity of this eval grid (with pad)
            rfft_size = (Nx + pad) // 2 + 1
            if K > rfft_size:
                log(f'  skip N={Nx+1}: K={K} > rfft_size={rfft_size}'); continue
            Phi, e = P.get_phi(Nx)
            U0 = np.stack([LG.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
            NU = P.make_nu(Nx, xi, Phi, e); Fds = P.downsample(F513, Nx)
            PT = build_pt_eig(Nx, Nt, P.AD, P.SDG)
            for kind in ['FrFNO', 'FNO']:
                key = f'N64_K{K}_{Nx+1}_{kind}'
                if key in E: continue
                er = P.per_sample_err(nets[kind], 'A' if kind=='FrFNO' else 'F', Nx, U0, NU, a, s, Fds, PT)
                E[key] = er; np.savez(EVAL_NPZ, E=E)
                log(f'  [eval] {key}: mean {er.mean():.3f}%')

    # Evaluate baseline K=10,N0=17^2 at N=33 (for Exp1, if not already in p0_eval)
    bw = os.path.join(OUT, 'kscan_K10.pt')
    if os.path.exists(bw):
        nets, MEAN, SV, _ = load_nets(bw, 10, pad=4)
        SC.MEAN, SC.SV = MEAN, SV
        Nx, Nt = 32, 3200
        Phi, e = P.get_phi(Nx)
        U0 = np.stack([LG.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
        NU = P.make_nu(Nx, xi, Phi, e); Fds = P.downsample(F513, Nx)
        PT = build_pt_eig(Nx, Nt, P.AD, P.SDG)
        for kind in ['FrFNO', 'FNO']:
            key = f'N17_K10_{Nx+1}_{kind}'
            if key in E: continue
            er = P.per_sample_err(nets[kind], 'A' if kind=='FrFNO' else 'F', Nx, U0, NU, a, s, Fds, PT)
            E[key] = er; np.savez(EVAL_NPZ, E=E)
            log(f'  [eval] {key}: mean {er.mean():.3f}%')
    else:
        log('baseline kscan_K10.pt not found, Exp1 uses published data from summary')

    np.savez(EVAL_NPZ, E=E); log('eval DONE')

@torch.no_grad()
def stage_spec():
    """Exp2: residual spectral tail sigma_K(r) using DST basis (matches theory), unnormalized.

    Theory (Prop. reg): sigma_K(f)^2 = ||(I-P_K)f||^2, P_K truncates DST modes with |(i,j)|>=K.
    We verify sigma_K(r)/sigma_K(u) <= C K^{-2s} using the SAME unnormalized DST tail energy.
    """
    from gpu_solver import dst2_torch
    zr = nz(REF_NPZ)
    if not zr or zr.get('n_done', 0) < P.NTE: log('ref not ready'); return
    F513, seeds, a, s, xi = zr['F'], zr['seeds'], zr['a'], zr['s'], zr['xi']
    zs = nz(SPEC_NPZ); SPEC = zs['SPEC'].item() if zs and 'SPEC' in zs else {}

    Nx = 64  # training resolution
    Phi, e = P.get_phi(Nx)
    U0 = np.stack([LG.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
    NU = P.make_nu(Nx, xi, Phi, e)
    Fds = P.downsample(F513, Nx)  # (B, (Nx+1)^2)
    PT = build_pt_eig(Nx, 3200, P.AD, P.SDG)

    # DST mode indices i,j = 1..Nx; radial cutoff K means sqrt(i^2+j^2) >= K
    ii, jj = np.meshgrid(np.arange(1, Nx+1), np.arange(1, Nx+1), indexing='ij')
    kr = np.sqrt(ii**2 + jj**2)

    Ks = [4, 6, 8, 12]
    all_sigma_r = {K: [] for K in Ks}
    all_sigma_u = {K: [] for K in Ks}
    svals = []
    for i in range(len(a)):
        aa, ss = float(a[i]), float(s[i])
        g = prop_at(aa, ss, *PT)
        base = ub_full_batch(U0[i:i+1], g, Nx).cpu().numpy()[0]  # (N+1,N+1)
        u_ref = Fds[i].reshape(Nx+1, Nx+1)
        r = u_ref - base
        # DST on interior (Nx, Nx)
        r_int = torch.tensor(r[1:, 1:], device=DEV, dtype=DT).unsqueeze(0)
        u_int = torch.tensor(u_ref[1:, 1:], device=DEV, dtype=DT).unsqueeze(0)
        r_dst = dst2_torch(r_int)[0].cpu().numpy()  # (Nx, Nx), ortho -> Parseval holds
        u_dst = dst2_torch(u_int)[0].cpu().numpy()
        pr = r_dst**2; pu = u_dst**2
        for K in Ks:
            mask = kr >= K
            all_sigma_r[K].append(np.sqrt(pr[mask].sum()))
            all_sigma_u[K].append(np.sqrt(pu[mask].sum()))
        svals.append(ss)

    for K in Ks:
        sr = np.array(all_sigma_r[K]); su = np.array(all_sigma_u[K])
        key = f'N64_K{K}'
        SPEC[key] = {'sigma_r_mean': float(sr.mean()), 'sigma_r_std': float(sr.std()),
                     'sigma_u_mean': float(su.mean()), 'sigma_u_std': float(su.std()),
                     'ratio_mean': float((sr/su).mean()),
                     's_mean': float(np.mean(svals)), 'radial_K': K}
        np.savez(SPEC_NPZ, SPEC=SPEC)
        log(f'[spec] K={K}: sigma_r={sr.mean():.4e} sigma_u={su.mean():.4e} ratio={(sr/su).mean():.4e} (s_mean={np.mean(svals):.3f})')
    log('spec DONE')

def stage_plot():
    """Generate all three figures."""
    # ===== Exp1: N->inf convergence curve =====
    # Published data from exp5_both_summary.md (baseline K=10, N0=17^2)
    Ns_pub = [17, 65, 129, 257]
    frfno_pub = [33.12, 12.21, 10.50, 10.58]
    fno_pub = [33.10, 19.98, 20.34, 21.06]
    # Check if we have N=33 from our eval
    ze = nz(EVAL_NPZ)
    if ze and 'E' in ze:
        E = ze['E'].item()
        v33_f = E.get('N17_K10_33_FrFNO')
        v33_n = E.get('N17_K10_33_FNO')
        if v33_f is not None:
            Ns_pub = [17, 33, 65, 129, 257]
            frfno_pub = [33.12, float(v33_f.mean()), 12.21, 10.50, 10.58]
            fno_pub = [33.10, float(v33_n.mean()), 19.98, 20.34, 21.06]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(Ns_pub, frfno_pub, '-o', color='#D9534F', lw=2, ms=7, label='FrFNO')
    ax.plot(Ns_pub, fno_pub, '--s', color='#1F6FB2', lw=2, ms=7, label='FNO')
    ax.axhline(y=frfno_pub[-1], color='#D9534F', ls=':', alpha=0.5, label=f'FrFNO floor={frfno_pub[-1]:.1f}%')
    ax.axhline(y=fno_pub[-1], color='#1F6FB2', ls=':', alpha=0.5, label=f'FNO floor={fno_pub[-1]:.1f}%')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('evaluation grid $N$ (points per side)'); ax.set_ylabel('rel.$L^2$ error (%)')
    ax.set_title('Exp.1: $N\\to\\infty$ convergence to a constant error floor ($K{=}10$, $N_0{=}17^2$)')
    ax.legend(fontsize=8); ax.grid(alpha=.3, which='both'); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'exp5_conv_curve.png'), dpi=150); plt.close()
    log(f'Exp1 plot saved: Ns={Ns_pub}, FrFNO floor={frfno_pub[-1]:.2f}%, FNO floor={fno_pub[-1]:.2f}%')

    # ===== Exp2: spectral tail ratio sigma_r/sigma_u =====
    zs = nz(SPEC_NPZ)
    if zs and 'SPEC' in zs:
        SPEC = zs['SPEC'].item()
        Ks = sorted([K for K in [4,6,8,12] if f'N64_K{K}' in SPEC])
        if len(Ks) >= 3:
            sr = np.array([SPEC[f'N64_K{K}']['sigma_r_mean'] for K in Ks])
            su = np.array([SPEC[f'N64_K{K}']['sigma_u_mean'] for K in Ks])
            ratio = sr / su
            Ks_arr = np.array(Ks, dtype=float)
            slope_r, _ = np.polyfit(np.log10(Ks_arr), np.log10(sr), 1)
            slope_u, _ = np.polyfit(np.log10(Ks_arr), np.log10(su), 1)
            slope_ratio, _ = np.polyfit(np.log10(Ks_arr), np.log10(ratio), 1)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.loglog(Ks_arr, sr, 'o-', color='#C0392B', ms=8, lw=2, label=f'$\\sigma_K(r)$ slope={slope_r:.2f}')
            ax.loglog(Ks_arr, su, 's-', color='#2980B9', ms=8, lw=2, label=f'$\\sigma_K(u)$ slope={slope_u:.2f}')
            ax.loglog(Ks_arr, ratio, '^-', color='#2E8B57', ms=8, lw=2, label=f'$\\sigma_K(r)/\\sigma_K(u)$ slope={slope_ratio:.2f}')
            ax.set_xlabel('radial modes $K$'); ax.set_ylabel('normalized spectral tail')
            ax.set_title('Exp.~5b: residual vs solution spectral tail ($N_0{=}65^2$, true residual)')
            ax.legend(fontsize=9); ax.grid(alpha=.3, which='both'); plt.tight_layout()
            plt.savefig(os.path.join(OUT, 'exp5_spec_tail.png'), dpi=150); plt.close()
            log(f'Exp2 plot saved: slope_r={slope_r:.3f} slope_u={slope_u:.3f} slope_ratio={slope_ratio:.3f}')
        else:
            log('Exp2: not enough spec data points')
    else:
        log('Exp2: no spec data, run stage_spec first')

    # ===== Exp3: K-as-bottleneck floor comparison =====
    if ze and 'E' in ze:
        E = ze['E'].item()
        Ks_n64 = []; floor_frfno_n64 = []; floor_fno_n64 = []
        for K, _ in N64_CONFIGS:
            vf = E.get(f'N64_K{K}_257_FrFNO')
            vn = E.get(f'N64_K{K}_257_FNO')
            if vf is not None and vn is not None:
                Ks_n64.append(K); floor_frfno_n64.append(float(vf.mean())); floor_fno_n64.append(float(vn.mean()))
        # N0=17^2 K-scan (published data, N0 is bottleneck)
        Ks_n17 = [10, 16, 24]
        frfno_n17 = [10.58, 14.80, 18.52]
        fno_n17 = [21.06, 48.33, 51.65]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
        # Left: N0=65^2 (K is bottleneck)
        if len(Ks_n64) >= 2:
            ax1.plot(Ks_n64, floor_frfno_n64, '-o', color='#D9534F', lw=2, ms=8, label='FrFNO')
            ax1.plot(Ks_n64, floor_fno_n64, '--s', color='#1F6FB2', lw=2, ms=8, label='FNO')
        ax1.set_xlabel('spectral modes $K$'); ax1.set_ylabel('error floor at $N{=}257$ (%)')
        ax1.set_title('$N_0{=}65^2$: $K$ is bottleneck\nfloor decreases with $K$')
        ax1.legend(fontsize=9); ax1.grid(alpha=.3)
        # Right: N0=17^2 (N0 is bottleneck)
        ax2.plot(Ks_n17, frfno_n17, '-o', color='#D9534F', lw=2, ms=8, label='FrFNO')
        ax2.plot(Ks_n17, fno_n17, '--s', color='#1F6FB2', lw=2, ms=8, label='FNO')
        ax2.set_xlabel('spectral modes $K$'); ax2.set_ylabel('error floor at $N{=}257$ (%)')
        ax2.set_title('$N_0{=}17^2$: $N_0$ is bottleneck\nfloor increases with $K$ (overfitting)')
        ax2.legend(fontsize=9); ax2.grid(alpha=.3)
        plt.suptitle('Exp.3: error floor controlled by $\\min(K,N_0)$', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, 'exp5_k_bottleneck.png'), dpi=150, bbox_inches='tight'); plt.close()
        log(f'Exp3 plot saved: N64 Ks={Ks_n64}, FrFNO floors={floor_frfno_n64}')
    else:
        log('Exp3: no eval data, run stage_eval first')

    log('all plots DONE')

if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if st in ('eval', 'all'): stage_eval()
    if st in ('spec', 'all'): stage_spec()
    if st in ('plot', 'all'): stage_plot()
    print('EXP5 SUPPLEMENT', st, 'DONE')
