# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SEC = _os.path.dirname(_HERE); _PKG_ROOT = _os.path.dirname(_SEC)
for _p in (_HERE, _SEC, _PKG_ROOT):
    if _p not in _sys.path: _sys.path.insert(0, _p)
"""P0-1 strict error floor + K scan (Proposition 3, prediction iv, both directions).
ref:    a single ultra-high-resolution ground truth at 513^2 / Nt=6400 (nonlinear beta=3), saved sample by sample and resumable.
eval:   evaluation tiers 17/65/129/257; predictions are inferred at each tier, and the ground truth is uniformly downsampled from 513 by local averaging (removing the effect that the ground truth sharpens with N).
        weights K=10/16/24 (kscan_K*.pt); per-sample relL2 saved.
summary: 1000-bootstrap 95% CI for the mean, plateau tests of adjacent-tier increments, and an N-error curve (as K grows, the floor should move monotonically down)."""
import os, sys, time
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import frfno_legacy as LG
from frfno_legacy import DualNet, DEV, DT, prop_at, ub_full_batch
from gpu_solver import spectral_eigvals_torch
from nl_solver import gpu_solve_nl_batch
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
import surrogate_compare as SC
from theory_verify import build_pt_eig

OUT = os.path.join(ROOT, 'theory_closure')
TN, BETA = 0.015, 3.0; LG.T = TN; SC.T = TN
NX_REF, NT_REF = 512, 6400
RES = [(16, 3200), (64, 3200), (128, 3200), (256, 3200)]   # evaluation tiers; the propagator uniformly uses Nt=3200 (matched dt accuracy)
KS = [10, 16, 24]
NTE = 48
AD = np.linspace(0.45, 1.05, 21); SDG = np.linspace(0.25, 0.90, 21)
REF_NPZ = os.path.join(OUT, 'p0_ref513.npz'); EVAL_NPZ = os.path.join(OUT, 'p0_eval.npz')
L = []
def log(s): print(s, flush=True); L.append(str(s)); open(os.path.join(OUT, 'p0_platform.log'), 'w', encoding='utf-8').write('\n'.join(L))
def nz(p): return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else {}

def sample_design():
    rng = np.random.RandomState(2024); seeds = rng.randint(100000, size=NTE)
    a = rng.uniform(0.55, 0.95, NTE).astype(np.float32)
    s = rng.uniform(0.35, 0.78, NTE).astype(np.float32)
    xi = rng.randn(NTE, KMAX).astype(np.float32)
    return seeds, a, s, xi

def get_phi(Nx):
    if (Nx, 'Phi') not in PHI_CACHE:
        p, e = kl_basis(Nx, KMAX); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = p, e
    return PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')]

# ---------- Stage 1: 513 ground truth, saved sample by sample ----------
@torch.no_grad()
def stage_ref():
    seeds, a, s, xi = sample_design()
    z = nz(REF_NPZ); Fdone = z['F'] if z and 'F' in z else None
    n0 = 0 if Fdone is None else len(Fdone)
    if n0 >= NTE: log('ref already done'); return
    Phi, e = get_phi(NX_REF); Q = spectral_eigvals_torch(NX_REF, DEV, DT)
    Fs = [] if Fdone is None else [Fdone[k] for k in range(n0)]
    for i in range(n0, NTE):
        t0 = time.time()
        U0 = np.stack([LG.generate_multiscale_initial(NX_REF, 8, int(seeds[i]))]).astype(np.float32)
        NU = make_nu(NX_REF, xi[i:i+1], Phi, e)
        f = gpu_solve_nl_batch(U0, NU, a[i:i+1], s[i:i+1], TN, NT_REF, NX_REF, beta=BETA,
                               Q=Q, device=DEV, dtype=DT).cpu().numpy()[0]
        Fs.append(f)
        np.savez(REF_NPZ, F=np.array(Fs), seeds=seeds, a=a, s=s, xi=xi, n_done=i+1)
        log(f'[ref] {i+1}/{NTE} single sample {time.time()-t0:.1f}s')
    log('ref ALL DONE')

def downsample(F513, N):
    """Locally average a 513^2 field (B,513,513) down to (N+1)^2; 512/N must divide evenly."""
    B = F513.shape[0]; q = 512//N; inner = F513[:, 1:513, 1:513].reshape(B, N, q, N, q).mean(axis=(2, 4))
    out = np.zeros((B, N+1, N+1), np.float32); out[:, 1:N+1, 1:N+1] = inner
    return out.reshape(B, -1)

@torch.no_grad()
def per_sample_err(net, kind, Nx, U0, NU, a, s, Ftrue_ds, PT):
    errs = []
    for i in range(len(a)):
        u0t = torch.tensor(U0[i:i+1], device=DEV, dtype=DT); nut = torch.tensor(NU[i:i+1], device=DEV, dtype=DT)
        aa, ss = float(a[i]), float(s[i])
        if kind == 'A':
            gs = [prop_at(aa, ss, *PT)]+[prop_at(*zz, *PT) for zz in [(aa-0.10, ss), (aa+0.10, ss), (aa, ss-0.08), (aa, ss+0.08)]]
            ubs = [ub_full_batch(U0[i:i+1], g, Nx) for g in gs]
            xb = torch.stack([u0t, ubs[0], nut, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            phys = ubs[0].cpu().numpy()+SC.SV*net(xb, (aa, ss)).cpu().numpy()
        else:
            xb = torch.stack([u0t, nut], -1)
            phys = SC.MEAN+SC.SV*net(xb, (aa, ss)).cpu().numpy()
        errs.append(np.linalg.norm(phys.reshape(-1)-Ftrue_ds[i])/(np.linalg.norm(Ftrue_ds[i])+1e-12))
    return 100*np.array(errs)

# ---------- Stage 2: per-sample error for each tier x K x method ----------
@torch.no_grad()
def stage_eval():
    zr = nz(REF_NPZ)
    if not zr or zr['n_done'] < NTE: log('ref not finished, cannot eval'); return
    F513 = zr['F']; seeds, a, s, xi = zr['seeds'], zr['a'], zr['s'], zr['xi']
    st = nz(kscan_trainstats_path()) if os.path.exists(kscan_trainstats_path()) else {}
    SC.MEAN = float(st.get('MEAN', 0.0051)); SC.SV = float(st.get('SV', 0.2266))
    ze = nz(EVAL_NPZ); E = ze['E'].item() if ze and 'E' in ze else {}
    for K in KS:
        wpath = os.path.join(OUT, f'kscan_K{K}.pt')
        if not os.path.exists(wpath): log(f'missing weights kscan_K{K}.pt, skip this K'); continue
        ck = torch.load(wpath, map_location=DEV); pad = int(ck['pad'])
        nets = {}
        nets['FrFNO'] = DualNet(7, modes=K, pad=pad, field_dim=1, seed=1).to(DEV).float(); nets['FrFNO'].load_state_dict(ck['FrFNO']); nets['FrFNO'].eval()
        nets['FNO'] = DualNet(2, modes=K, pad=pad, field_dim=0, seed=2).to(DEV).float(); nets['FNO'].load_state_dict(ck['FNO']); nets['FNO'].eval()
        for Nx, Nt in RES:
            Phi, e = get_phi(Nx)
            U0 = np.stack([LG.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
            NU = make_nu(Nx, xi, Phi, e)
            Fds = downsample(F513, Nx)
            t0 = time.time(); PT = build_pt_eig(Nx, Nt, AD, SDG)
            for kind in ['FrFNO', 'FNO']:
                key = f'K{K}_{Nx+1}_{kind}'
                if key in E: continue
                er = per_sample_err(nets[kind], 'A' if kind == 'FrFNO' else 'F', Nx, U0, NU, a, s, Fds, PT)
                E[key] = er
                np.savez(EVAL_NPZ, E=E); log(f'[eval] {key}: mean {er.mean():.3f}% median {np.median(er):.3f}%')
            log(f'  (N={Nx+1} table build + eval {time.time()-t0:.0f}s)')
    np.savez(EVAL_NPZ, E=E); log('eval ALL DONE')

def kscan_trainstats_path(): return os.path.join(OUT, 'kscan_trainstats.npz')

# ---------- Stage 3: bootstrap CI + plateau/K curves ----------
def stage_summary():
    ze = nz(EVAL_NPZ)
    if not ze or 'E' not in ze: log('eval not finished'); return
    E = ze['E'].item(); rng = np.random.RandomState(0); NB = 2000
    Ns = [Nx+1 for Nx, _ in RES]
    def boot(v):
        idx = rng.randint(0, len(v), (NB, len(v))); m = v[idx].mean(1)
        return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
    summ = {}
    log('\n===== strict plateau (uniform evaluation by downsampling one 513 ground truth, 48 samples, bootstrap 95% CI) =====')
    log('N^2    ' + ''.join(f'{m:>22}' for m in ['FrFNO(K10)','FNO(K10)','FrFNO(K16)','FNO(K16)','FrFNO(K24)','FNO(K24)']))
    for N in Ns:
        row = []
        for K in KS:
            for kind in ['FrFNO', 'FNO']:
                v = E.get(f'K{K}_{N}_{kind}')
                row.append(boot(v) if v is not None else (np.nan,)*3); summ[(K, N, kind)] = row[-1]
        def fmt(t): return f'{t[0]:6.3f}[{t[1]:.3f},{t[2]:.3f}]'
        log(f'{N:<5} ' + ''.join(f'{fmt(t):>22}' for t in row))
    # Plateau increment test: whether the adjacent-tier difference under K10 crosses 0 (CI)
    log('\n--- plateau increment test (FrFNO K10: later minus earlier tier mean difference, bootstrap CI; containing 0 means a statistical plateau) ---')
    for kind in ['FrFNO', 'FNO']:
        for j in range(1, len(Ns)):
            v1 = E[f'K10_{Ns[j-1]}_{kind}']; v2 = E[f'K10_{Ns[j]}_{kind}']
            d = (v2[None]-v1[None] + np.zeros((NB, 1))).ravel()
            # paired bootstrap
            n = len(v1); idx = rng.randint(0, n, (NB, n)); dd = (v2[idx]-v1[idx]).mean(1)
            log(f'  {kind} {Ns[j-1]}->{Ns[j]}: Δ={dd.mean():+.3f}% CI[{np.percentile(dd,2.5):+.3f},{np.percentile(dd,97.5):+.3f}] {"plateau (contains 0)" if np.percentile(dd,2.5)<=0<=np.percentile(dd,97.5) else "still significant change"}')
    # plot: N-error, one FrFNO (solid)/FNO (dashed) curve per K
    plt.figure(figsize=(6.6,4.6)); cmap = {10:'#D9534F', 16:'#1F6FB2', 24:'#2E8B57'}
    for K in KS:
        for kind, ls in [('FrFNO','-o'), ('FNO','--s')]:
            ys = [summ[(K, N, kind)][0] for N in Ns]; lo = [summ[(K,N,kind)][1] for N in Ns]; hi = [summ[(K,N,kind)][2] for N in Ns]
            plt.plot(Ns, ys, ls, color=cmap[K], lw=2, ms=6, label=f'{kind} K={K}')
            plt.fill_between(Ns, lo, hi, color=cmap[K], alpha=.12)
    plt.xscale('log'); plt.yscale('log'); plt.xlabel('grid $N$ (points per side)'); plt.ylabel('rel.L2 error (%)')
    plt.title('Prop.3(iv): N-plateau at fixed K; widening K alone (fixed $17^2$ train) does not lower floor'); plt.legend(fontsize=8); plt.grid(alpha=.3, which='both'); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'p0_floor_Kscan.png'), dpi=150); plt.close()
    np.savez(os.path.join(OUT, 'p0_summary.npz'),
             **{f'K{K}_N{N}_{k}': np.array(summ[(K,N,k)]) for K in KS for N in Ns for k in ['FrFNO','FNO']})
    log('summary DONE -> p0_floor_Kscan.png / p0_summary.npz')

if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('ref', 'all'): stage_ref()
    if stage in ('eval', 'all'): stage_eval()
    if stage in ('summary', 'all'): stage_summary()
    print('P0 STAGE', stage, 'DONE')
