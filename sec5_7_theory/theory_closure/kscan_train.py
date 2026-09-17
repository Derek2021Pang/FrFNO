# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SEC = _os.path.dirname(_HERE); _PKG_ROOT = _os.path.dirname(_SEC)
for _p in (_HERE, _SEC, _PKG_ROOT):
    if _p not in _sys.path: _sys.path.insert(0, _p)
"""P0-1 (K scan) retraining: fix the training grid at 17^2 and nonlinear beta=3 (B1 protocol); vary only the number of spectral modes K and the matching pad.
K=10/16/24 -> pad=4/8/16 (ensures the padded spectral grid has >=2K points per axis and rfft dimension >=K). Train FrFNO+FNO for each K,
write kscan_K{K}.pt to disk immediately, and skip if it already exists, so the run is resumable."""
import os, sys, time
ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)
import numpy as np, torch
import frfno_legacy as LG
from frfno_legacy import DualNet, build_prop_table, DEV
import surrogate_compare as SC
import b1_burgers as B1
from models_surrogate import n_params

OUT = os.path.join(ROOT, 'theory_closure')
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, 'kscan_train.log')
L = []
def log(s): print(s, flush=True); L.append(str(s)); open(LOG, 'w', encoding='utf-8').write('\n'.join(L))

KC = {10: 4, 16: 15, 24: 31}    # K -> pad; pad is added once on the right/bottom; H'=17+pad must satisfy >=2K and H'//2+1>=K
STEPS = 30 if os.environ.get('FAST') == '1' else 8000

def main():
    LG.T = B1.TN; SC.T = B1.TN; SC.STEPS = STEPS; SC.NTR = B1.NTR
    for Nx in (16,):
        from d_scan import kl_basis, KMAX, PHI_CACHE
        Phi, e = kl_basis(16, KMAX); PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')] = Phi, e
    U0tr, SOL, XItrain, HIST = B1.build_trainset_nl()
    SC.MEAN = float(SOL.mean()); SC.SV = float(SOL.std())
    log(f'SOL stats MEAN={SC.MEAN:.4f} SV={SC.SV:.4f} STEPS={STEPS}')
    PT16 = build_prop_table(16, Nt=B1.NL_NT)
    np.savez(os.path.join(OUT, 'kscan_trainstats.npz'), MEAN=SC.MEAN, SV=SC.SV)
    for K, pad in KC.items():
        wpath = os.path.join(OUT, f'kscan_K{K}.pt')
        if os.path.exists(wpath):
            log(f'K={K} already exists, skip'); continue
        tK = time.time(); ck = {'K': K, 'pad': pad}
        for name, kind, seed, fd in [('FrFNO', 'A', 1, 1), ('FNO', 'F', 2, 0)]:
            net = DualNet(7 if kind == 'A' else 2, modes=K, pad=pad, field_dim=fd, seed=seed).to(DEV).float()
            tt = SC.train_spectral(net, kind, SOL, U0tr, XItrain, PT16, f'K{K}-{name}')
            ck[name] = net.state_dict(); ck[f'{name}_nparam_K'] = n_params(net)/1e3; ck[f'{name}_t'] = tt
            torch.save(ck, wpath)            # write to disk immediately after each finishes
            log(f'K={K} {name}: params {n_params(net)/1e3:.1f}K train {tt:.0f}s')
        torch.save(ck, wpath)
        log(f'K={K} done in {time.time()-tK:.0f}s')
    log('ALL K TRAIN DONE')

if __name__ == '__main__':
    main()
