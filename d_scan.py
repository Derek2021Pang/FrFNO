# -*- coding: utf-8 -*-
"""
KL basis and diffusivity field generation utilities for FrFNO surrogate modeling.
Provides:
  - kl_basis: Karhunen-Loeve basis for random diffusivity fields
  - make_nu: generate log-normal diffusivity fields from KL coefficients
  - KMAX, PHI_CACHE: spectral cache utilities
"""
import os, sys, time
from math import comb
import numpy as np
import torch
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from numpy.polynomial.hermite_e import hermegauss
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import frfno_core as U
from frfno_core import (DualNet, build_prop_table, prop_at, ub_full_batch, tn_ppf,
                           discrete_gauss_1d, err_of, mom, wmom, coverage, RNG_A, RNG_S,
                           generate_multiscale_initial, DEV, DT, T, DA, DS, A_GRID, S_GRID)
from gpu_solver import gpu_solve_imex_batch, spectral_eigvals_torch
import torch.nn.functional as F

NX_EVAL, NT_EVAL = 64, 800          # target resolution
NX_COARSE = 32                       # MLQMC coarse level
DNU_LIST = [0, 3, 8, 18]             # -> d = 2,5,10,20
KMAX = 20                            # KL full modes (used in training)
BETA, SIGG = 1.0, 0.35               # spectral decay (roughness) / pointwise std of log-nu
FAST = os.environ.get('FAST') == '1'
NTR = 16 if FAST else 128
STEPS = 30 if FAST else 8000
NREF = 32 if FAST else 256           # MC reference (Sobol, power of 2)
NTE = 2 if FAST else 2
MINF = 32 if FAST else 128           # neural-operator inference samples
MAXNODE = 1500                       # gPC node cap; beyond it flag as infeasible
MLC_NC, MLC_NF = (16, 8) if FAST else (128, 48)   # MLQMC coarse/fine samples
OUT = os.path.join(ROOT, 'd_scan_result.txt')
L = []
def log(s): print(s, flush=True); L.append(str(s))
def save(): open(OUT, 'w', encoding='utf-8').write('\n'.join(L))

# ============ High-dimensional KL log-normal random field (cosine spectral basis, analytic basis consistent with resolution) ============
def kl_basis(Nx, kmax=KMAX, beta=BETA):
    xs = np.linspace(0, 1, Nx+1); X, Y = np.meshgrid(xs, xs, indexing='ij')
    cand = [(m, n) for m in range(8) for n in range(8)]
    cand = sorted(cand, key=lambda mn: 1+mn[0]**2+mn[1]**2)[:kmax]
    Phi = np.zeros((kmax, Nx+1, Nx+1), np.float32); e = np.zeros(kmax)
    for k, (m, n) in enumerate(cand):
        phi = np.cos(np.pi*m*X)*np.cos(np.pi*n*Y)
        phi = phi/np.sqrt((phi**2).mean())
        Phi[k] = phi.astype(np.float32); e[k] = (1+m**2+n**2)**(-beta)
    e = e/e.sum()
    return Phi, e.astype(np.float32)

def nu_from_xi(xi, Phi, e, sig=SIGG):
    B = len(xi); _, H, W = Phi.shape; dnu = xi.shape[1]; g = np.zeros((B, H, W), np.float32)
    for k in range(dnu):
        g += sig*np.sqrt(e[k])*xi[:, k][:, None, None]*Phi[k][None]
    nu = np.exp(g); return (nu/nu.mean(axis=(1, 2), keepdims=True)).astype(np.float32)

# ============ Nested Leja rule on the unified CDF space [0,1] + L2 Smolyak ============
def leja_seq(n_max=9, n_cand=4001):
    g = np.linspace(0, 1, n_cand); chosen = [0.5]; d = np.abs(g-0.5)
    while len(chosen) < n_max:
        i = np.argmax(d); chosen.append(float(g[i])); d *= np.abs(g-g[i])
    return np.array(chosen)
LEJA = leja_seq()
def lagrange_w(nodes, nq=4001):
    q = np.linspace(0, 1, nq); w = np.zeros(len(nodes))
    for j, xj in enumerate(nodes):
        lj = np.ones_like(q)
        for m, xm in enumerate(nodes):
            if m != j: lj *= (q-xm)/(xj-xm)
        w[j] = np.trapz(lj, q)
    return w
def smolyak_L2(D):
    """Nested Leja at level 2: 1+2D unique nodes; u in [0,1]^D; returns the nodes u and the combination weights."""
    p1 = LEJA[:1]; w1 = lagrange_w(p1)
    p2 = LEJA[:3]; w2 = lagrange_w(p2)
    dw = np.zeros(3); dw[0] = w2[0]-w1[0]; dw[1:] = w2[1:]
    U = np.full((1, D), 0.5); w = np.array([1.0])
    for j in range(D):
        for t in (1, 2):
            row = np.full((1, D), 0.5); row[0, j] = p2[t]
            U = np.vstack([U, row]); w = np.append(w, dw[t])
    return U.astype(np.float64), w.astype(np.float64)
def u_to_z(U):
    """CDF space -> (a,s,xi); column 0 -> a, column 1 -> s, remaining columns -> standard-normal xi."""
    a = tn_ppf(U[:, 0], RNG_A); s = tn_ppf(U[:, 1], RNG_S)
    xi = norm.ppf(np.clip(U[:, 2:], 1e-10, 1-1e-10)) if U.shape[1] > 2 else np.zeros((len(U), 0))
    return a, s, xi
def sobol_z(Dtot, n, seed=42):
    sb = Sobol(d=Dtot, scramble=True, seed=seed).random(n)
    a = tn_ppf(sb[:, 0], RNG_A); s = tn_ppf(sb[:, 1], RNG_S)
    xi = norm.ppf(np.clip(sb[:, 2:], 1e-10, 1-1e-10)) if Dtot > 2 else np.zeros((n, 0))
    return a, s, xi
def gpc_z(dnu, na=3, ns=3, nh=3):
    if na*ns*nh**dnu > MAXNODE:
        return None
    xa, wa = discrete_gauss_1d(RNG_A, na); xs, ws = discrete_gauss_1d(RNG_S, ns)
    xh, wh = hermegauss(nh); wh = wh/np.sqrt(2*np.pi)
    grids = [xa, xs]+[xh]*dnu; wgrids = [wa, ws]+[wh]*dnu
    mg = np.meshgrid(*grids, indexing='ij'); mw = np.meshgrid(*wgrids, indexing='ij')
    a = mg[0].ravel(); s = mg[1].ravel()
    xi = np.stack([mg[2+k].ravel() for k in range(dnu)], 1) if dnu else np.zeros((a.size, 0))
    w = np.ones(a.size)
    for q in mw: w *= q.ravel()
    return a, s, xi, w

# ============ Batched solve (given a,s,xi -> flattened fields) ============
def solve_batch(Nx, Nt, u0, a, s, xi, bs=32):
    Q = spectral_eigvals_torch(Nx, DEV, DT); outs = []; t0 = time.time()
    Phi, e = PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')]
    for i0 in range(0, len(a), bs):
        xb = xi[i0:i0+bs]
        nub = make_nu(Nx, xb, Phi, e)
        o = gpu_solve_imex_batch(np.repeat(u0[None], len(xb), 0), nub,
                                 a[i0:i0+bs], s[i0:i0+bs], T, Nt, Nx, Q=Q,
                                 device=DEV, dtype=DT).cpu().numpy()
        outs.append(o.reshape(len(xb), -1))
    return np.concatenate(outs, 0), (time.time()-t0)/len(a)
PHI_CACHE = {}
def make_nu(Nx, xi, Phi, e):
    B = len(xi); _, H, W = Phi.shape; g = np.zeros((B, H, W), np.float32)
    for k in range(xi.shape[1]):
        g += SIGG*np.sqrt(e[k])*xi[:, k][:, None, None]*Phi[k][None]
    nu = np.exp(g); return (nu/nu.mean(axis=(1, 2), keepdims=True)).astype(np.float32)

def interp_to_fine(Fc, Nc, Nf, shape_f):
    t = torch.tensor(Fc.reshape(-1, Nc+1, Nc+1)[:, None], device=DEV, dtype=DT)
    t = torch.nn.functional.interpolate(t, size=(Nf+1, Nf+1), mode='bilinear', align_corners=False)
    return t[:, 0].reshape(len(Fc), -1).cpu().numpy()

# ============ Train FrFNO/Naive (full-mode KMAX fields, train once, infer for all d) ============
SV = MEAN = None
def build_trainset():
    log(f'\n=== 17^2 training data, {NTR} fields x 9x9 order grid; nu=full-mode KL (K={KMAX}) ===')
    Phi, e = kl_basis(16, KMAX); PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')] = Phi, e
    rng = np.random.RandomState(0)
    U0 = np.stack([generate_multiscale_initial(16, 8, 1000+i) for i in range(NTR)]).astype(np.float32)
    XI = rng.randn(81, NTR, KMAX).astype(np.float32).reshape(9, 9, NTR, KMAX)
    AA = np.repeat(A_GRID, 9*NTR); SS = np.tile(np.repeat(S_GRID, NTR), 9)
    NU = make_nu(16, XI.reshape(81*NTR, KMAX), Phi, e)
    UB = np.concatenate([U0]*81, 0)
    Q = spectral_eigvals_torch(16, DEV, DT); outs = []; t0 = time.time()
    for i0 in range(0, len(AA), 1024):
        outs.append(gpu_solve_imex_batch(UB[i0:i0+1024], NU[i0:i0+1024], AA[i0:i0+1024],
                                         SS[i0:i0+1024], T, 400, 16, Q=Q,
                                         device=DEV, dtype=DT).cpu().numpy())
    SOL = np.concatenate(outs, 0).reshape(9, 9, NTR, 17, 17)
    log(f'  training ground truth {9*9*NTR}  samples {time.time()-t0:.0f}s')
    return U0, SOL, XI
def train(net, kind, SOL, U0, XI, AD, SD, PT, tag):
    import torch.optim as opt
    op = opt.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5)
    sch = opt.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5); net.train(); t0 = time.time()
    Phi, e = PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')]
    for st in range(STEPS+1):
        itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
        ia, js = np.random.randint(9), np.random.randint(9); a, s = A_GRID[ia], S_GRID[js]
        sol = torch.tensor(SOL[ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0[itr], device=DEV, dtype=DT)
        xi = XI[ia, js, itr]; nub = torch.tensor(make_nu(16, xi, Phi, e), device=DEV, dtype=DT)
        if kind == 'A':
            gs = [prop_at(a, s, AD, SD, PT)]+[prop_at(*z, AD, SD, PT) for z in
                  [(a-DA, s), (a+DA, s), (a, s-DS), (a, s+DS)]]
            ubs = [ub_full_batch(U0[itr], g, 16) for g in gs]
            xb = torch.stack([u0b, ubs[0], nub, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            tg = (sol-ubs[0])/SV
        else:
            xb = torch.stack([u0b, nub], -1); tg = (sol-MEAN)/SV
        loss = F.mse_loss(net(xb, (a, s)), tg)
        op.zero_grad(); loss.backward(); op.step(); sch.step()
    net.eval(); log(f'  [{tag}] {time.time()-t0:.0f}s'); return time.time()-t0

@torch.no_grad()
def infer(net, kind, u0, a, s, xi, Nx, AD, SD, PT):
    """Forward each random realization (different a,s,xi) one sample at a time; batch=1 field -> (M,D) flattened physics fields."""
    Phi, e = PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')]
    per = []
    u0t = torch.tensor(u0[None], device=DEV, dtype=DT)
    for m in range(len(a)):
        nu = make_nu(Nx, xi[m:m+1], Phi, e); nut = torch.tensor(nu, device=DEV, dtype=DT)
        aa, ss = float(a[m]), float(s[m])
        if kind == 'A':
            gs = [prop_at(aa, ss, AD, SD, PT)]+[prop_at(*z, AD, SD, PT) for z in
                  [(aa-DA, ss), (aa+DA, ss), (aa, ss-DS), (aa, ss+DS)]]
            ubs = [ub_full_batch(u0[None], g, Nx) for g in gs]
            xb = torch.stack([u0t, ubs[0], nut, ubs[1], ubs[2], ubs[3], ubs[4]], -1)
            o = net(xb, (aa, ss)).cpu().numpy(); phys = ubs[0].cpu().numpy()+SV*o
        else:
            xb = torch.stack([u0t, nut], -1)
            o = net(xb, (aa, ss)).cpu().numpy(); phys = MEAN+SV*o
        per.append(phys.reshape(-1))
    return np.stack(per)

# ============ Main pipeline ============
def main():
    global SV, MEAN
    for Nx in (16, NX_COARSE, NX_EVAL):
        Phi, e = kl_basis(Nx, KMAX); PHI_CACHE[(Nx, 'Phi')], PHI_CACHE[(Nx, 'e')] = Phi, e
    U0tr, SOL, XItrain = build_trainset(); SV = SOL.std(); MEAN = SOL.mean()
    ADtr, SDtr, PTtr = build_prop_table(16, Nt=400)
    netA = DualNet(7, seed=1); tA = train(netA, 'A', SOL, U0tr, XItrain, ADtr, SDtr, PTtr, 'FrFNO')
    netB = DualNet(2, seed=10); tB = train(netB, 'B', SOL, U0tr, XItrain, ADtr, SDtr, PTtr, 'Naive')
    AD, SD, PT = build_prop_table(NX_EVAL, Nt=400)
    ADC, SDC, PTC = build_prop_table(NX_COARSE, Nt=400)
    log(f'\nOne-time training: FrFNO {tA:.0f}s, Naive {tB:.0f}s (full-mode KL, shared for all d)')

    table = {}
    for dnu in DNU_LIST:
        Dtot = 2+dnu; log(f'\n================ dnu={dnu}  total stochastic dimension d={Dtot} ================')
        agg = {}
        for it in range(NTE):
            u0 = generate_multiscale_initial(NX_EVAL, 8, 500000+it).astype(np.float32)
            u0c = generate_multiscale_initial(NX_COARSE, 8, 500000+it).astype(np.float32)
            # --- MC reference (Sobol master sequence, nested QMC) ---
            ra, rs, rxi = sobol_z(Dtot, NREF, seed=100+it)
            Fref, cs = solve_batch(NX_EVAL, NT_EVAL, u0, ra, rs, rxi)
            mr, vr = mom(Fref); res = {}
            for nq in (32, 64, 128):
                if nq <= NREF:
                    m, v = mom(Fref[:nq]); res[f'QMC{nq}'] = (*err_of(m, v, mr, vr), coverage(m, v, Fref), nq*cs)
            # --- gPC tensor product (check node count before building, to avoid high-d meshgrid memory blow-up) ---
            ng = 9*3**dnu
            gpc = gpc_z(dnu)
            if gpc is not None:
                ga, gs, gxi, gw = gpc
                Fg, cg = solve_batch(NX_EVAL, NT_EVAL, u0, ga, gs, gxi)
                m, v = wmom(Fg, gw); res['gPC'] = (*err_of(m, v, mr, vr), coverage(m, v, Fref), ng*cg)
            else:
                res['gPC'] = (np.nan, np.nan, np.nan, np.nan, np.nan); log(f'  gPC tensor nodes {ng} > {MAXNODE}, flag infeasible (curse of dimensionality)')
            # --- SC nested Leja-Smolyak L2 ---
            Us, ws = smolyak_L2(Dtot); ca, cs2, cxi = u_to_z(Us)
            Fs, csc = solve_batch(NX_EVAL, NT_EVAL, u0, ca, cs2, cxi)
            m, v = wmom(Fs, ws); res['SC-L2'] = (*err_of(m, v, mr, vr), coverage(m, v, Fref), len(ca)*csc)
            # --- Two-level MLQMC (33^2 coarse + 65^2 fine), same Sobol sequence ---
            tml0 = time.time()
            za, zs, zxi = sobol_z(Dtot, MLC_NC, seed=100+it)
            Fc_raw, _ = solve_batch(NX_COARSE, NT_EVAL//2, u0c, za, zs, zxi)   # coarse level, multiple samples
            Fci_all = interp_to_fine(Fc_raw, NX_COARSE, NX_EVAL, (NX_EVAL+1)**2)  # interpolate all coarse solutions onto the fine grid
            Ff, _ = solve_batch(NX_EVAL, NT_EVAL, u0, za[:MLC_NF], zs[:MLC_NF], zxi[:MLC_NF])
            Fci_nf = Fci_all[:MLC_NF]
            Fd = Ff-Fci_nf                                                    # same-point coarse-fine difference (for the first-moment level difference)
            mc, _ = mom(Fci_all)
            mml = mc + Fd.mean(0)                                            # multilevel estimate of the first moment
            e2c = (Fci_all**2).mean(0)
            e2fine = e2c + ((Ff**2).mean(0)-(Fci_nf**2).mean(0))            # multilevel estimate of the second moment (level diff = E[Ff^2]-E[Fc^2])
            vml = np.maximum(e2fine-mml**2, 0)
            cml = time.time()-tml0
            res['MLQMC'] = (*err_of(mml, vml, mr, vr), coverage(mml, vml, Fref), cml)
            # --- FrFNO / Naive (forward MINF random realizations, independent of d) ---
            ia_, is_, ixi = sobol_z(Dtot, MINF, seed=2024+it)
            t0 = time.time(); pA = infer(netA, 'A', u0, ia_, is_, ixi, NX_EVAL, AD, SD, PT); tAinf = time.time()-t0
            m, v = mom(pA); res['FrFNO'] = (*err_of(m, v, mr, vr), coverage(m, v, Fref), tAinf)
            t0 = time.time(); pB = infer(netB, 'B', u0, ia_, is_, ixi, NX_EVAL, AD, SD, PT); tBinf = time.time()-t0
            m, v = mom(pB); res['Naive'] = (*err_of(m, v, mr, vr), coverage(m, v, Fref), tBinf)
            for k in res: agg.setdefault(k, []).append(res[k])
        log(f'  {"Method":<8}{"meanerr":>10}{"std err":>10}{"std corr":>9}{"cover":>8}{"cost/s":>9}')
        row = {}
        for k, vals in agg.items():
            arr = np.array(vals, float); ok = ~np.isnan(arr[:, 0]); row[k] = arr[ok].mean(0) if ok.any() else arr[0]
            a = row[k]
            if np.isnan(a[0]): log(f'  {k:<8}   infeasible (node explosion)')
            else: log(f'  {k:<8}{a[0]*100:>9.3f}%{a[1]*100:>9.3f}%{a[2]:>9.4f}{a[3]*100:>7.1f}%{a[4]:>9.2f}')
        table[Dtot] = row; save()
    # summary: mean/std error and cost vs d
    methods = ['QMC32', 'QMC64', 'QMC128', 'gPC', 'SC-L2', 'MLQMC', 'Naive', 'FrFNO']
    log('\n========= summary: mean err% vs stochastic dimension d =========')
    log('d   ' + ''.join(f'{m:>10}' for m in methods))
    for Dtot, row in table.items():
        line = f'{Dtot:<4}'
        for m in methods:
            v = row[m][0] if m in row else np.nan
            line += ('   INF    ' if np.isnan(v) else f'{v*100:>9.3f}%')
        log(line)
    log('========= summary: std err% vs stochastic dimension d =========')
    log('d   ' + ''.join(f'{m:>10}' for m in methods))
    for Dtot, row in table.items():
        line = f'{Dtot:<4}'
        for m in methods:
            v = row[m][1] if m in row else np.nan
            line += ('   INF    ' if np.isnan(v) else f'{v*100:>9.3f}%')
        log(line)
    log('========= summary: target-resolution cost s vs d =========')
    log('d   ' + ''.join(f'{m:>10}' for m in methods))
    for Dtot, row in table.items():
        line = f'{Dtot:<4}'
        for m in methods:
            v = row[m][4] if m in row else np.nan
            line += ('   INF    ' if np.isnan(v) else f'{v:>10.2f}')
        log(line)
    np.savez(os.path.join(ROOT, 'd_scan_metrics.npz'),
             table=np.array([[table[d][m] for m in methods] for d in sorted(table)], object),
             methods=np.array(methods), ds=np.array(sorted(table)))
    save()

if __name__ == '__main__':
    main()
