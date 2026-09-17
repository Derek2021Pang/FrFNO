# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SEC = _os.path.dirname(_HERE); _PKG_ROOT = _os.path.dirname(_SEC)
for _p in (_HERE, _SEC, _PKG_ROOT):
    if _p not in _sys.path: _sys.path.insert(0, _p)
"""Exp5 supplementary evaluation: K and training-grid enlarged TOGETHER.
Evaluates (K=16,Nx_tr=32) and (K=24,Nx_tr=64) on the same frozen 513^2 truth,
and compares with only-K-widened and only-grid-refined baselines.
Tests Corollary cor:plateau: floor drops only when K AND training bandwidth grow together.

Usage: python p0_both_eval.py [eval|summary|all]
"""
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

OUT = P.OUT
RES = P.RES  # [(16,3200),(64,3200),(128,3200),(256,3200)]
EVAL_NPZ = os.path.join(OUT, 'p0_both_eval.npz')
L = []
def log(s): print(s, flush=True); L.append(str(s)); open(os.path.join(OUT, 'p0_both_eval.log'), 'w', encoding='utf-8').write('\n'.join(L))

# Configuration table: label -> (K, Nx_tr, weight_file, is_both)
CONFIGS = [
    ('K10_N16 (baseline)',     10, 16, os.path.join(OUT, 'kscan_K10.pt'),      False),
    ('K16_N16 (K only)',       16, 16, os.path.join(OUT, 'kscan_K16.pt'),      False),
    ('K24_N16 (K only)',       24, 16, os.path.join(OUT, 'kscan_K24.pt'),      False),
    ('K10_N32 (grid only)',    10, 32, os.path.join(OUT, 'trNx32_K10.pt'),     False),
    ('K10_N64 (grid only)',    10, 64, os.path.join(OUT, 'trNx64_K10.pt'),     False),
    ('K16_N32 (BOTH)',         16, 32, os.path.join(OUT, 'both_K16_Nx32.pt'),  True),
    ('K24_N64 (BOTH)',         24, 64, os.path.join(OUT, 'both_K24_Nx64.pt'),  True),
]

KC = {10: 4, 16: 15, 24: 31}

def load_pair(label, K, Nx_tr, wpath):
    ck = torch.load(wpath, map_location=DEV)
    MEAN = float(ck.get('MEAN', 0.0051)); SV = float(ck.get('SV', 0.2266))
    pad = int(ck.get('pad', KC[K]))
    nets = {}
    nets['FrFNO'] = DualNet(7, modes=K, pad=pad, field_dim=1, seed=1).to(DEV).float()
    nets['FrFNO'].load_state_dict(ck['FrFNO']); nets['FrFNO'].eval()
    nets['FNO'] = DualNet(2, modes=K, pad=pad, field_dim=0, seed=2).to(DEV).float()
    nets['FNO'].load_state_dict(ck['FNO']); nets['FNO'].eval()
    log(f'  loaded {label}: K={K}, Nx_tr={Nx_tr}, pad={pad}, MEAN={MEAN:.4f}, SV={SV:.4f}')
    return nets, MEAN, SV

@torch.no_grad()
def stage_eval():
    zr = P.nz(P.REF_NPZ)
    if not zr or zr['n_done'] < P.NTE: log('513 truth not ready'); return
    F513, seeds, a, s, xi = zr['F'], zr['seeds'], zr['a'], zr['s'], zr['xi']
    ze = P.nz(EVAL_NPZ); E = ze['E'].item() if ze and 'E' in ze else {}
    for label, K, Nx_tr, wpath, is_both in CONFIGS:
        if not os.path.exists(wpath): log(f'  MISSING {wpath}, skip'); continue
        nets, MEAN, SV = load_pair(label, K, Nx_tr, wpath)
        SC.MEAN, SC.SV = MEAN, SV
        for Nx, Nt in RES:
            Phi, e = P.get_phi(Nx)
            import frfno_legacy as LG
            U0 = np.stack([LG.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
            NU = P.make_nu(Nx, xi, Phi, e); Fds = P.downsample(F513, Nx)
            PT = build_pt_eig(Nx, Nt, P.AD, P.SDG)
            for kind in ['FrFNO', 'FNO']:
                key = f'{label}_{Nx+1}_{kind}'
                if key in E: continue
                er = P.per_sample_err(nets[kind], 'A' if kind=='FrFNO' else 'F', Nx, U0, NU, a, s, Fds, PT)
                E[key] = er; np.savez(EVAL_NPZ, E=E)
                log(f'  [eval] {key}: mean {er.mean():.3f}% median {np.median(er):.3f}%')
    np.savez(EVAL_NPZ, E=E); log('eval DONE')

def stage_summary():
    ze = P.nz(EVAL_NPZ)
    if not ze or 'E' not in ze: log('eval not done'); return
    E = ze['E'].item(); rng = np.random.RandomState(1); NB = 2000
    Ns = [Nx+1 for Nx, _ in RES]
    def boot(v):
        idx = rng.randint(0, len(v), (NB, len(v))); m = v[idx].mean(1)
        return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
    log('\n' + '='*90)
    log('Exp5 SUPPLEMENTARY: K AND TRAINING GRID ENLARGED TOGETHER')
    log('(single frozen 513^2 truth, 48 samples, bootstrap 95% CI)')
    log('='*90)
    for kind in ['FrFNO', 'FNO']:
        log(f'\n--- {kind} ---')
        log(f'{"configuration":<28} ' + ' '.join(f'N={n:>5}' for n in Ns))
        for label, K, Nx_tr, wpath, is_both in CONFIGS:
            row = []
            for N in Ns:
                v = E.get(f'{label}_{N}_{kind}')
                row.append(boot(v) if v is not None else (np.nan,)*3)
            marker = ' *' if is_both else ''
            log(f'{label+marker:<28} ' + ' '.join(f'{t[0]:7.3f}' for t in row))
    # Key comparison at N=257 (plateau region)
    log('\n' + '='*90)
    log('KEY COMPARISON at N=257 (plateau region):')
    log('='*90)
    for kind in ['FrFNO', 'FNO']:
        log(f'\n{kind}:')
        vals = {}
        for label, K, Nx_tr, wpath, is_both in CONFIGS:
            v = E.get(f'{label}_257_{kind}')
            if v is not None: vals[label] = v.mean()
        for label, v in vals.items():
            log(f'  {label:<28} {v:.3f}%')
        # Compare: K-only vs grid-only vs both
        if 'K16_N16 (K only)' in vals and 'K10_N32 (grid only)' in vals and 'K16_N32 (BOTH)' in vals:
            k16 = vals['K16_N16 (K only)']; g32 = vals['K10_N32 (grid only)']; b16 = vals['K16_N32 (BOTH)']
            log(f'\n  K=16 axis: K-only={k16:.2f}%, grid-only={g32:.2f}%, BOTH={b16:.2f}%')
            log(f'  BOTH vs grid-only: {b16-g32:+.2f}% ({"lower" if b16<g32 else "NOT lower"})')
        if 'K24_N16 (K only)' in vals and 'K10_N64 (grid only)' in vals and 'K24_N64 (BOTH)' in vals:
            k24 = vals['K24_N16 (K only)']; g64 = vals['K10_N64 (grid only)']; b24 = vals['K24_N64 (BOTH)']
            log(f'\n  K=24 axis: K-only={k24:.2f}%, grid-only={g64:.2f}%, BOTH={b24:.2f}%')
            log(f'  BOTH vs grid-only: {b24-g64:+.2f}% ({"lower" if b24<g64 else "NOT lower"})')
    log('\nsummary DONE')

if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if st in ('eval', 'all'): stage_eval()
    if st in ('summary', 'all'): stage_summary()
    print('BOTH EVAL', st, 'DONE')
