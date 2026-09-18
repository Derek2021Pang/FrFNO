# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SEC = _os.path.dirname(_HERE); _PKG_ROOT = _os.path.dirname(_SEC)
for _p in (_HERE, _SEC, _PKG_ROOT):
    if _p not in _sys.path: _sys.path.insert(0, _p)
"""P0-1 training (physical) resolution axis: fix K=10/pad=4, training grids Nx_tr=16/32/64 (17/33/65),
and compare error floors on evaluation tiers downsampled from the same 513 ground truth. Nx_tr=16 reuses kscan_K10.pt.
evaltr saves per-sample relL2; summarytr gives the bootstrap CI and the training-resolution-axis curve."""
import os, sys
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'theory_closure'))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from frfno_legacy import DualNet, DEV
import surrogate_compare as SC
from theory_verify import build_pt_eig
import p0_platform as P

OUT = P.OUT; RES = P.RES; AD, SDG = P.AD, P.SDG
TR_GRID = [16, 32, 64]
EVAL_NPZ = os.path.join(OUT, 'p0_tr_eval.npz')
L = []
def log(s): print(s, flush=True); L.append(str(s)); open(os.path.join(OUT, 'p0_trscan.log'), 'w', encoding='utf-8').write('\n'.join(L))

def wpath(Ntr):
    if Ntr == 16: return os.path.join(OUT, 'kscan_K10.pt')
    return os.path.join(OUT, f'trNx{Ntr}_K10.pt')

def load_pair(Ntr):
    wp = wpath(Ntr); ck = torch.load(wp, map_location=DEV)
    if Ntr == 16:
        st = P.nz(os.path.join(OUT, 'kscan_trainstats.npz')); MEAN, SV = float(st['MEAN']), float(st['SV']); pad = int(ck['pad'])
    else:
        MEAN, SV, pad = float(ck['MEAN']), float(ck['SV']), int(ck['pad'])
    nets = {}
    nets['FrFNO'] = DualNet(7, modes=10, pad=pad, field_dim=1, seed=1).to(DEV).float(); nets['FrFNO'].load_state_dict(ck['FrFNO']); nets['FrFNO'].eval()
    nets['FNO'] = DualNet(2, modes=10, pad=pad, field_dim=0, seed=2).to(DEV).float(); nets['FNO'].load_state_dict(ck['FNO']); nets['FNO'].eval()
    return nets, MEAN, SV

@torch.no_grad()
def stage_evaltr():
    zr = P.nz(P.REF_NPZ)
    if not zr or zr['n_done'] < P.NTE: log('513 ground truth not finished, waiting'); return
    F513, seeds, a, s, xi = zr['F'], zr['seeds'], zr['a'], zr['s'], zr['xi']
    ze = P.nz(EVAL_NPZ); E = ze['E'].item() if ze and 'E' in ze else {}
    for Ntr in TR_GRID:
        if not os.path.exists(wpath(Ntr)): log(f'missing weights Nx_tr={Ntr}, skip'); continue
        nets, MEAN, SV = load_pair(Ntr); SC.MEAN, SC.SV = MEAN, SV
        for Nx, Nt in RES:
            Phi, e = P.get_phi(Nx)
            import frfno_legacy as LG
            U0 = np.stack([LG.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
            NU = P.make_nu(Nx, xi, Phi, e); Fds = P.downsample(F513, Nx)
            PT = build_pt_eig(Nx, Nt, AD, SDG)
            for kind in ['FrFNO', 'FNO']:
                key = f'tr{Ntr}_{Nx+1}_{kind}'
                if key in E: continue
                er = P.per_sample_err(nets[kind], 'A' if kind == 'FrFNO' else 'F', Nx, U0, NU, a, s, Fds, PT)
                E[key] = er; np.savez(EVAL_NPZ, E=E); log(f'[evaltr] {key}: mean {er.mean():.3f}% median {np.median(er):.3f}% (MEAN={MEAN:.4f},SV={SV:.4f})')
    np.savez(EVAL_NPZ, E=E); log('evaltr DONE')

def stage_summarytr():
    ze = P.nz(EVAL_NPZ)
    if not ze or 'E' not in ze: log('evaltr not finished'); return
    E = ze['E'].item(); rng = np.random.RandomState(1); NB = 2000
    Ns = [Nx+1 for Nx, _ in RES]
    def boot(v):
        idx = rng.randint(0, len(v), (NB, len(v))); m = v[idx].mean(1); return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
    summ = {}
    log('\n===== training-resolution axis (fixed K=10; single 513 ground truth; 48 samples, bootstrap 95% CI) =====')
    for kind in ['FrFNO', 'FNO']:
        log(f'--- {kind} ---  ' + ''.join(f'{n:>22}' for n in Ns))
        for Ntr in TR_GRID:
            row = []
            for N in Ns:
                v = E.get(f'tr{Ntr}_{N}_{kind}')
                if v is None: row.append((np.nan,)*3)
                else: row.append(boot(v)); summ[(kind, Ntr, N)] = row[-1]
            log(f'train {Ntr+1 if False else Ntr:>2}:  ' + ''.join(f'{t[0]:7.3f}[{t[1]:.2f},{t[2]:.2f}]'.rjust(22) for t in row))
    cmap = {16: '#D9534F', 32: '#1F6FB2', 64: '#2E8B57'}
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for j, kind in enumerate(['FrFNO', 'FNO']):
        for Ntr in TR_GRID:
            ys = [summ.get((kind, Ntr, N), (np.nan,)*3)[0] for N in Ns]
            lo = [summ.get((kind, Ntr, N), (np.nan,)*3)[1] for N in Ns]; hi = [summ.get((kind, Ntr, N), (np.nan,)*3)[2] for N in Ns]
            ax[j].plot(Ns, ys, '-o', color=cmap[Ntr], lw=2, label=f'train {Ntr+1}$^2$')
            ax[j].fill_between(Ns, lo, hi, color=cmap[Ntr], alpha=.12)
        ax[j].set_xscale('log'); ax[j].set_yscale('log'); ax[j].set_xlabel('eval grid $N$'); ax[j].set_ylabel('rel.L2 (%)')
        ax[j].set_title(f'{kind}: floor vs training resolution (K=10)'); ax[j].legend(); ax[j].grid(alpha=.3, which='both')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'p0_floor_trainres.png'), dpi=150); plt.close()
    np.savez(os.path.join(OUT, 'p0_tr_summary.npz'),
             **{f'{k}_tr{Ntr}_N{N}': np.array(summ[(k, Ntr, N)]) for (k, Ntr, N) in summ})
    log('summarytr DONE -> p0_floor_trainres.png')

if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if st in ('evaltr', 'all'): stage_evaltr()
    if st in ('summarytr', 'all'): stage_summarytr()
    print('TR STAGE', st, 'DONE')
