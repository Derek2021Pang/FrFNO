# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
B2 long-horizon (T=0.06) component ablation, mirroring the B1 C1 ablation table
(c_ablation.py) so the two can be placed side by side in the paper.

Six variants (single-knife, matched backbone ~1880K where applicable):
  full     : 7 explicit channels [u0, u_base^(a,s), nu, 4 neighbor bases], cond ON, DynFrac fd=2, learn RESIDUAL
  no_ub    : 2 channels [u0, nu], no analytic base at all, cond ON, fd=2, learn FULL FIELD (MEAN baseline)
  no_dyn   : 7 channels, cond ON, DynFrac fd=0 (dynamic fractional feature off), residual
  no_nb    : 3 channels [u0, u_base^(a,s), nu], neighbor bases removed, cond ON, fd=2, residual
  direct   : 7 channels, cond ON, fd=2, learn FULL FIELD instead of residual
  no_cond  : 7 channels, cond OFF (alpha/s constant fields removed), fd=2, residual  [NEW: isolates conditioning]

Mid-to-upper scale: NTR=128 fields/order, STEPS=5000, NTEST=32, 9x9 orders, three eval tiers.
Incremental checkpointing into b2_ablation.npz allows resume; results streamed to txt.
"""
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = _PKG_ROOT
sys.path.insert(0, ROOT)

from frfno_core import (SpecConv2d, DynFrac2d, build_prop_table,
                         A_GRID, S_GRID, DEV, DT)
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
from surrogate_compare import prop_at, ub_full_batch, DA, DSNB
import b2b_longT as B2

# ---------------- mid-to-upper protocol ----------------
FAST = os.environ.get('FAST') == '1'
NTR = 8 if FAST else 128
STEPS = 4 if FAST else 5000
NTEST = 3 if FAST else 32
BETA = 3.0
TN = 0.06
NL_NT_TRAIN = 1600
PT_NT = 40 if FAST else 1600
RES_TEST = [(16, 40), (64, 80), (128, 160)] if FAST else [(16, 1600), (64, 3200), (128, 6400)]
LR = 1.5e-3
WD = 1e-5
SEED_NET = 0

OUT_TXT = os.path.join(ROOT, 'b2_ablation_result' + ('_fast' if FAST else '') + '.txt')
OUT_NPZ = os.path.join(ROOT, 'b2_ablation' + ('_fast' if FAST else '') + '.npz')
L = []
def log(s): print(s, flush=True); L.append(str(s)); open(OUT_TXT, 'w', encoding='utf-8').write('\n'.join(L))
def dump(d):
    z = dict(np.load(OUT_NPZ, allow_pickle=True)) if os.path.exists(OUT_NPZ) else {}
    z.update(d); np.savez(OUT_NPZ, **z)
def done_keys():
    if not os.path.exists(OUT_NPZ): return set()
    z = np.load(OUT_NPZ, allow_pickle=True)
    return set(json.loads(z['done'].item())) if 'done' in z.files else set()
def save_done(k):
    z = dict(np.load(OUT_NPZ, allow_pickle=True)) if os.path.exists(OUT_NPZ) else {}
    s = set(json.loads(z['done'].item()) if 'done' in z else []); s.add(k)
    z['done'] = json.dumps(sorted(s)); np.savez(OUT_NPZ, **z)


# ---------------- network with switchable conditioning (same as core DualNet otherwise) ----------------
class DualNetCond(nn.Module):
    def __init__(self, in_dim, modes=10, width=48, n_layers=4, field_dim=2,
                 pad=4, emb=32, seed=0, use_cond=True):
        super().__init__()
        torch.manual_seed(seed); self.pad = pad; self.nl = n_layers
        self.act = nn.LeakyReLU(); self.use_cond = use_cond
        self.frac = DynFrac2d(field_dim, modes)
        lifted = in_dim + 2 + self.frac.extra + (2 if use_cond else 0)
        self.fc0 = nn.Sequential(nn.Linear(lifted, 128), self.act, nn.Linear(128, width))
        self.sp = nn.ModuleList([SpecConv2d(width, width, modes) for _ in range(n_layers)])
        self.cv = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.enc = nn.Sequential(nn.Linear(2, emb), nn.GELU(), nn.Linear(emb, emb))
        self.fg = nn.ModuleList([nn.Linear(emb, width) for _ in range(n_layers)])
        self.fb = nn.ModuleList([nn.Linear(emb, width) for _ in range(n_layers)])
        self.q = nn.Sequential(nn.Linear(width, 128), self.act, nn.Linear(128, 1))
        self.to(DEV)

    def forward(self, x, cond):
        a, s = float(cond[0]), float(cond[1]); B, H, W, _ = x.shape
        gx = torch.linspace(0, 1, W, device=x.device).view(1, 1, W, 1).repeat(B, H, 1, 1)
        gy = torch.linspace(0, 1, H, device=x.device).view(1, H, 1, 1).repeat(B, 1, W, 1)
        x = torch.cat([torch.cat([gy, gx], -1), x], -1)
        h = self.frac(x, a)
        if self.use_cond:
            ach = torch.cat([torch.full((B, H, W, 1), a, device=x.device),
                             torch.full((B, H, W, 1), s, device=x.device)], -1)
            h = torch.cat([h, ach], -1)
        h = self.fc0(h).permute(0, 3, 1, 2)
        e = self.enc(torch.tensor([[a, s]], device=DEV, dtype=DT).repeat(B, 1))
        if self.pad > 0: h = F.pad(h, [0, self.pad, 0, self.pad])
        for i in range(self.nl):
            h = self.sp[i](h) + self.cv[i](h)
            g = (1 + self.fg[i](e)).unsqueeze(-1).unsqueeze(-1)
            b = self.fb[i](e).unsqueeze(-1).unsqueeze(-1)
            h = g * h + b
            if i != self.nl - 1: h = self.act(h)
        if self.pad > 0: h = h[..., :-self.pad, :-self.pad]
        return self.q(h.permute(0, 2, 3, 1)).squeeze(-1)


def n_params(net): return sum(p.numel() for p in net.parameters())


# ---------------- variant specification ----------------
# in_dim: explicit input channels; fd: DynFrac field_dim; cond: alpha/s constant fields;
# resid: True -> learn (sol - central base), False -> learn full field (MEAN baseline)
VARIANTS = [
    # key,      in_dim, fd, cond,  resid
    ('full',    7,      2,  True,  True),
    ('no_ub',   2,      2,  True,  False),
    ('no_dyn',  7,      0,  True,  True),
    ('no_nb',   3,      2,  True,  True),
    ('direct',  7,      2,  True,  False),
    ('no_cond', 7,      2,  False, True),
]


def five_bases(U0np, a, s, PT, Nx):
    gs = [prop_at(a, s, *PT)] + [prop_at(*z, *PT) for z in
          [(a - DA, s), (a + DA, s), (a, s - DSNB), (a, s + DSNB)]]
    return [ub_full_batch(U0np, g, Nx) for g in gs]


def build_input(key, u0b, nub, U0np, a, s, PT, Nx):
    """Return (xb, central_base or None)."""
    if key == 'no_ub':
        return torch.stack([u0b, nub], -1), None
    if key == 'no_nb':
        g = prop_at(a, s, *PT); ub = ub_full_batch(U0np, g, Nx)
        return torch.stack([u0b, ub, nub], -1), ub
    # full / no_dyn / direct / no_cond all see the 7 channels
    ubs = five_bases(U0np, a, s, PT, Nx)
    xb = torch.stack([u0b, ubs[0], nub, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
    return xb, ubs[0]


def train_one(key, in_dim, fd, use_cond, resid, SOL, U0, XI, PT16, MEAN, SV):
    net = DualNetCond(in_dim, modes=10, width=48, n_layers=4, field_dim=fd,
                      pad=4, emb=32, seed=SEED_NET, use_cond=use_cond)
    op = torch.optim.Adam(net.parameters(), LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5)
    net.train(); t0 = time.time()
    rng = np.random.RandomState(123)
    Phi, e = PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')]
    for st in range(STEPS + 1):
        ia, js = rng.randint(9), rng.randint(9)
        itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
        a, s = A_GRID[ia], S_GRID[js]
        sol = torch.tensor(SOL[ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0[itr], device=DEV, dtype=DT)
        nub = torch.tensor(make_nu(16, XI[ia, js, itr], Phi, e), device=DEV, dtype=DT)
        xb, ub0 = build_input(key, u0b, nub, U0[itr], a, s, PT16, 16)
        tg = (sol - ub0) / SV if resid else (sol - MEAN) / SV
        loss = F.mse_loss(net(xb, (a, s)), tg)
        op.zero_grad(); loss.backward(); op.step(); sch.step()
        if st % 1000 == 0:
            log('  [%s] step %d/%d  loss %.5f' % (key, st, STEPS, float(loss)))
    net.eval()
    return net, time.time() - t0


@torch.no_grad()
def eval_one(net, key, resid, Nx, U0, NU, a_te, s_te, Ftrue, PT, MEAN, SV):
    errs = []
    for i in range(len(a_te)):
        u0t = torch.tensor(U0[i:i + 1], device=DEV, dtype=DT)
        nut = torch.tensor(NU[i:i + 1], device=DEV, dtype=DT)
        aa, ss = float(a_te[i]), float(s_te[i])
        xb, ub0 = build_input(key, u0t, nut, U0[i:i + 1], aa, ss, PT, Nx)
        out = SV * net(xb, (aa, ss)).cpu().numpy()
        if resid:
            phys = ub0.cpu().numpy() + out
        else:
            phys = MEAN + out
        errs.append(np.linalg.norm(phys.reshape(-1) - Ftrue[i]) / (np.linalg.norm(Ftrue[i]) + 1e-12))
    return 100 * np.mean(errs), 100 * np.median(errs)


def main():
    log('=' * 78)
    log('B2 long-T (T=%g) component ablation, mid-upper scale' % TN)
    log('NTR=%d STEPS=%d NTEST=%d beta=%g NL_NT=%d, variants=%d' % (NTR, STEPS, NTEST, BETA, NL_NT_TRAIN, len(VARIANTS)))
    log('=' * 78)

    B2.NTR = NTR; B2.NL_NT = NL_NT_TRAIN; B2.TN = TN; B2.BETA = BETA
    for Nx, _ in RES_TEST:
        P, e = kl_basis(Nx, KMAX); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = P, e

    log('\n[1/4] Training ground truth (9x9 orders x %d fields, 17x17, T=%g)...' % (NTR, TN))
    t0 = time.time()
    U0tr, SOL, XI, _ = B2.build_trainset_nl()
    MEAN = float(SOL.mean()); SV = float(SOL.std())
    log('  %d samples, MEAN=%.4f SV=%.4f, %.1fs' % (81 * NTR, MEAN, SV, time.time() - t0))

    log('\n[2/4] Propagator tables (Nt=%d)...' % PT_NT)
    PT = {Nx: build_prop_table(Nx, Nt=PT_NT) for Nx, _ in RES_TEST}
    for Nx, _ in RES_TEST: log('  PT[%d] built' % Nx)

    log('\n[3/4] Shared test ground truth (%d samples x 3 tiers)...' % NTEST)
    rng = np.random.RandomState(2024)
    seeds_te = rng.randint(100000, size=NTEST)
    a_te = rng.uniform(A_GRID.min(), A_GRID.max(), NTEST).astype(np.float32)
    s_te = rng.uniform(S_GRID.min(), S_GRID.max(), NTEST).astype(np.float32)
    xi_te = rng.randn(NTEST, KMAX).astype(np.float32)
    TE = {}
    for Nx, Nt in RES_TEST:
        Fte, U0te, NUte = B2.true_field_nl(Nx, Nt, seeds_te, a_te, s_te, xi_te,
                                           PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')])
        TE[Nx] = (U0te, NUte, Fte); log('  test %dx%d done' % (Nx + 1, Nx + 1))
    dump(dict(MEAN=MEAN, SV=SV, NTR=NTR, STEPS=STEPS, NTEST=NTEST, TN=TN))

    log('\n[4/4] Train + evaluate each variant...')
    done = done_keys()
    rows = {}
    for key, in_dim, fd, use_cond, resid in VARIANTS:
        if key in done:
            log('  skip (done) ' + key); continue
        log('\n--- %s (in=%d, fd=%d, cond=%s, residual=%s) ---' % (key, in_dim, fd, use_cond, resid))
        net, tr = train_one(key, in_dim, fd, use_cond, resid, SOL, U0tr, XI, PT[16], MEAN, SV)
        log('  params %.1fK, train %.1fs' % (n_params(net) / 1e3, tr))
        e = {}
        for Nx, _ in RES_TEST:
            U0te, NUte, Fte = TE[Nx]
            em, emd = eval_one(net, key, resid, Nx, U0te, NUte, a_te, s_te, Fte, PT[Nx], MEAN, SV)
            e[Nx + 1] = round(float(em), 3)
            log('  %3dx%-3d mean=%7.3f%% median=%7.3f%%' % (Nx + 1, Nx + 1, em, emd))
        row = dict(key=key, in_dim=in_dim, fd=fd, cond=use_cond, resid=resid,
                   nparamK=round(n_params(net) / 1e3, 1), train_s=round(tr, 1), e=e)
        dump({key: json.dumps(row)}); save_done(key)
        rows[key] = row
        del net; torch.cuda.empty_cache()

    # ---------------- summary ----------------
    z = np.load(OUT_NPZ, allow_pickle=True)
    log('\n' + '=' * 78)
    log('SUMMARY  B2 T=%g  (rel.L2 %% , %d test fields, %d steps)' % (TN, NTEST, STEPS))
    log('=' * 78)
    log('%-9s %6s %6s %5s %8s | %9s %9s %9s' % ('variant', 'in_ch', 'fd', 'cond', 'paramsK', '17^2', '65^2', '129^2'))
    order = [v[0] for v in VARIANTS]
    full_e = None
    for key in order:
        if key not in z.files: continue
        r = json.loads(str(z[key].item()))
        if key == 'full': full_e = r['e']
        log('%-9s %6d %6d %5s %8.1f | %9.3f %9.3f %9.3f' % (
            r['key'], r['in_dim'], r['fd'], str(r['cond']), r['nparamK'], r['e']['17'], r['e']['65'], r['e']['129']))
    if full_e is not None:
        log('\nDelta vs full at 129^2 (positive = removing component HURTS):')
        for key in order:
            if key == 'full' or key not in z.files: continue
            r = json.loads(str(z[key].item()))
            d17 = r['e']['17'] - full_e['17']; d65 = r['e']['65'] - full_e['65']; d129 = r['e']['129'] - full_e['129']
            log('  %-9s 17^2: %+.2f   65^2: %+.2f   129^2: %+.2f pp' % (key, d17, d65, d129))
    log('\nSaved -> %s' % OUT_TXT)


if __name__ == '__main__':
    main()
