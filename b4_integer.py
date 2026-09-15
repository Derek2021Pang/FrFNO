# -*- coding: utf-8 -*-
"""B4 integer-order generality: the order grid is extended to the integer limit, focusing on the classical point (alpha,s)=(1,1)
(i.e. alpha=1 ordinary time derivative + s=1 classical Laplacian -> variable-coefficient viscous Burgers:
 du/dt + beta*u*d_xu = nu(x,y) Lap u), and on the near-integer continuous neighborhood [.9,1]^2.
Verifies that the FrFNO linear analytic propagator plus residual network remains valid and superior at the integer order.
Standalone process + incremental checkpoint (b4_ckpt.pt) + resume. FAST=1 gives a smoke test."""
import os, sys, time
os.environ.setdefault('FAST', '0')
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import frfno_core as FC
import d_scan as DS
from d_scan import kl_basis, make_nu, KMAX, PHI_CACHE
from frfno_core import DualNet, DEV, DT
from solver_spacetime_fractional import spectral_eigvals, _l1_b
from gpu_solver import spectral_eigvals_torch, dst2_torch
from nl_solver import gpu_solve_nl_batch, ddx_upwind
from models_surrogate import CNOWrap, DeepONet2d, UNet2d
from models_extra import PDNO2d
import surrogate_compare as SC

TN, BETA, NL_NT = 0.015, 3.0, 400
FC.T = TN
# Extend the order training grid to the integer-order limit 1.0
A_GRID = np.linspace(0.6, 1.0, 9).astype(np.float32)
S_GRID = np.linspace(0.4, 1.0, 9).astype(np.float32)
SC.A_GRID, SC.S_GRID = A_GRID, S_GRID          # override the grids internal to the training functions
MR, LAM = 32, 0.5
FAST = os.environ.get('FAST') == '1'
NTR = 16 if FAST else 128; STEPS = 30 if FAST else 8000; NTE = 8 if FAST else 48
RES = [(16, 400)] if FAST else [(16, 400), (64, 800), (128, 1600)]
METHODS = ['FrFNO', 'FNO', 'PINO', 'PDNO', 'CNO', 'DeepONet', 'UNet']
CKPT = os.path.join(ROOT, 'b4_ckpt.pt')
L = []
def log(s): print(s, flush=True); L.append(str(s))

def build_pt_b4(Nx, Nt=400, na=47, ns=43):
    """Propagator table whose parameter range covers the integer point (1,1) and the near-order extension (to 1.15/1.10)."""
    Q = spectral_eigvals(Nx); n = Nx-1
    AD = np.linspace(0.5, 1.15, na); SD = np.linspace(0.3, 1.10, ns)
    qf = torch.tensor(Q.reshape(-1), device=DEV, dtype=DT); PT = np.zeros((na, ns, n, n), np.float32)
    for ia, al in enumerate(AD):
        b = torch.tensor(_l1_b(al, Nt), device=DEV, dtype=DT)
        lam = qf[None] ** torch.tensor(SD, device=DEV, dtype=DT)[:, None]
        dta = (TN/Nt)**al; b0 = b[0]; hist = torch.empty((Nt, ns, n*n), device=DEV, dtype=DT); hist[0] = 1.0; v = hist[0]
        for k in range(1, Nt+1):
            rhs = b[k-1]*hist[0]
            if k > 1:
                jj = torch.arange(1, k, device=DEV); rhs = rhs + torch.einsum('j,jxy->xy', b[k-jj-1]-b[k-jj], hist[1:k])
            v = rhs/(b0+dta*lam)
            if k < Nt: hist[k] = v
        PT[ia] = v.reshape(ns, n, n).cpu().numpy(); del hist
    torch.cuda.empty_cache(); return AD, SD, PT

def build_trainset_nl():
    Phi, e = kl_basis(16, KMAX); PHI_CACHE[(16, 'Phi')], PHI_CACHE[(16, 'e')] = Phi, e
    rng = np.random.RandomState(0)
    U0 = np.stack([DS.generate_multiscale_initial(16, 8, 1000+i) for i in range(NTR)]).astype(np.float32)
    XI = rng.randn(81, NTR, KMAX).astype(np.float32).reshape(9, 9, NTR, KMAX)
    AA = np.repeat(A_GRID, 9*NTR); SS = np.tile(np.repeat(S_GRID, NTR), 9)
    NU = make_nu(16, XI.reshape(81*NTR, KMAX), Phi, e)
    UB = np.concatenate([U0]*81, 0); Q = spectral_eigvals_torch(16, DEV, DT); sols=[]; hists=[]; t0=time.time()
    for i0 in range(0, len(AA), 1024):
        fo, hb = gpu_solve_nl_batch(UB[i0:i0+1024], NU[i0:i0+1024], AA[i0:i0+1024], SS[i0:i0+1024],
                                    TN, NL_NT, 16, beta=BETA, Q=Q, device=DEV, dtype=DT, return_hist=True)
        sols.append(fo.cpu().numpy()); hists.append(hb.cpu().numpy())
    SOL = np.concatenate(sols, 0).reshape(9, 9, NTR, 17, 17)
    HIST = np.concatenate(hists, 0).reshape(9, 9, NTR, 17, 17)
    log('B4 training ground truth %d samples %.0fs (grid a %.2f-%.2f, s %.2f-%.2f)' % (9*9*NTR, time.time()-t0, A_GRID.min(), A_GRID.max(), S_GRID.min(), S_GRID.max()))
    return U0, SOL, XI, HIST

def true_field(Nx, Nt, seeds, a, s, xi, Phi, e):
    U0 = np.stack([DS.generate_multiscale_initial(Nx, 8, int(sd)) for sd in seeds]).astype(np.float32)
    NU = make_nu(Nx, xi, Phi, e); Q = spectral_eigvals_torch(Nx, DEV, DT)
    F = gpu_solve_nl_batch(U0, NU, a, s, TN, Nt, Nx, beta=BETA, Q=Q, device=DEV, dtype=DT).cpu().numpy()
    return F.reshape(len(a), -1), U0, NU

def train_pino(net, SOL, U0, XI, HIST, tag):
    import torch.optim as opt
    op = opt.Adam(net.parameters(), 1.5e-3, weight_decay=1e-5); sch = opt.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5)
    net.train(); rng = np.random.RandomState(123); Phi, e = PHI_CACHE[(16,'Phi')], PHI_CACHE[(16,'e')]; t0=time.time()
    Qmr = spectral_eigvals_torch(MR, DEV, DT)
    def up(z): return F.interpolate(z.unsqueeze(1), size=MR+1, mode='bicubic', align_corners=False).squeeze(1)
    for st in range(STEPS+1):
        ia, js = rng.randint(9), rng.randint(9); itr = torch.randperm(NTR, device=DEV)[:64].cpu().numpy()
        a, s = float(A_GRID[ia]), float(S_GRID[js])
        sol = torch.tensor(SOL[ia, js, itr], device=DEV, dtype=DT)
        u0b = torch.tensor(U0[itr], device=DEV, dtype=DT)
        nub = torch.tensor(make_nu(16, XI[ia, js, itr], Phi, e), device=DEV, dtype=DT)
        xb = torch.stack([u0b, nub], -1); ld = F.mse_loss(net(xb, (a, s)), (sol-SC.MEAN)/SC.SV)
        with torch.no_grad():
            u0f=up(u0b); nuf=up(nub); hf=up(torch.tensor(HIST[ia,js,itr],device=DEV,dtype=DT))
        pf=SC.MEAN+SC.SV*net(torch.stack([u0f,nuf],-1),(a,s)); pin=pf[:,1:-1,1:-1]
        dxu=ddx_upwind(pin,MR); fl=dst2_torch((Qmr**s)*dst2_torch(pin)); dta=(TN/NL_NT)**a
        r=pin-hf[:,1:-1,1:-1]+dta*(BETA*pin*dxu+nuf[:,1:-1,1:-1]*fl)
        lr=F.mse_loss(r,torch.zeros_like(r))/SC.SV**2; (ld+LAM*lr).backward(); op.step(); sch.step(); op.zero_grad()
    net.eval(); log('  [%-8s] train %ds params %.1fK'%(tag,time.time()-t0,sum(p.numel() for p in net.parameters())/1e3)); return time.time()-t0

def train_cno_fair(net, SOL, U0, XI, tag):
    """Official fair CNO recipe: use_bn=False + AdamW(1e-3,wd1e-8) + L1 + Cosine.
    The generic train_spatial Adam(1.5e-3,wd1e-5)+MSE diverges on grids that include the integer order s=1."""
    import torch.optim as opt
    op = opt.AdamW(net.parameters(), 1e-3, weight_decay=1e-8)
    sch = opt.lr_scheduler.CosineAnnealingLR(op, STEPS, 5e-5)
    net.train(); rng=np.random.RandomState(123); Phi,e=PHI_CACHE[(16,'Phi')],PHI_CACHE[(16,'e')]; t0=time.time()
    for st in range(STEPS+1):
        ia,js=rng.randint(9),rng.randint(9); itr=torch.randperm(NTR,device=DEV)[:64].cpu().numpy()
        a,s=float(A_GRID[ia]),float(S_GRID[js])
        sol=torch.tensor(SOL[ia,js,itr],device=DEV,dtype=DT)
        u0=torch.tensor(U0[itr],device=DEV,dtype=DT); nu=torch.tensor(make_nu(16,XI[ia,js,itr],Phi,e),device=DEV,dtype=DT)
        x=torch.stack([u0,nu],1); cond=torch.tensor(np.stack([np.full(len(itr),a),np.full(len(itr),s)],1),device=DEV,dtype=DT)
        loss=F.l1_loss(net(x,cond).unsqueeze(1),((sol-SC.MEAN)/SC.SV).unsqueeze(1))
        op.zero_grad(); loss.backward(); op.step(); sch.step()
    net.eval(); log('  [%-8s] train %ds params %.1fK (official AdamW+L1)'%(tag,time.time()-t0,sum(p.numel() for p in net.parameters())/1e3)); return time.time()-t0

def main():
    for Nx, _ in RES:
        Phi, e = kl_basis(Nx, KMAX); PHI_CACHE[(Nx,'Phi')], PHI_CACHE[(Nx,'e')] = Phi, e
    U0tr, SOL, XItr, HIST = build_trainset_nl()
    SC.MEAN=float(SOL.mean()); SC.SV=float(SOL.std()); SC.NTR=NTR; SC.STEPS=STEPS
    PT = {Nx: build_pt_b4(Nx, Nt) for Nx, Nt in RES}
    log('MEAN=%.4f SV=%.4f'%(SC.MEAN, SC.SV))
    ck = torch.load(CKPT, map_location=DEV) if os.path.exists(CKPT) else {}
    ck.pop('CNO', None)   # force retraining CNO with the official recipe (the old checkpoint used a non-official config and diverged)
    ck.pop('FNO', None)   # force retrain FNO with field_dim=2
    ck.pop('PINO', None)  # force retrain PINO with field_dim=2
    cons = {'FrFNO': lambda: DualNet(7, seed=1), 'FNO': lambda: DualNet(2, field_dim=2, seed=2),
            'PINO': lambda: DualNet(2, field_dim=2, seed=7), 'PDNO': lambda: PDNO2d(seed=6),
            'CNO': lambda: CNOWrap(17, ch=32, use_bn=False, seed=3),
            'DeepONet': lambda: DeepONet2d(w=256, seed=4), 'UNet': lambda: UNet2d(base=48, seed=5)}
    kind = {'FrFNO':'A','FNO':'F','PINO':'F','PDNO':'F'}; nets={}; tt={}
    for m in METHODS:
        net = cons[m]().to(DEV).float(); nets[m]=net
        if m in ck:
            net.load_state_dict(ck[m]); tt[m]=0.0; log('  restore %s skip training (restored)'%m); continue
        if m == 'PINO': tt[m]=train_pino(net, SOL, U0tr, XItr, HIST, m)
        elif m == 'CNO': tt[m]=train_cno_fair(net, SOL, U0tr, XItr, m)
        elif m in kind: tt[m]=SC.train_spectral(net, kind[m], SOL, U0tr, XItr, PT[16], m)
        else: tt[m]=SC.train_spatial(net, SOL, U0tr, XItr, m)
        ck[m]=net.state_dict(); torch.save(ck, CKPT)
    torch.save({m: nets[m].state_dict() for m in nets}, os.path.join(ROOT, 'b4_weights.pt'))

    def run_group(tag, afun):
        rng = np.random.RandomState(2024); seeds=rng.randint(100000,size=NTE); xit=rng.randn(NTE,KMAX).astype(np.float32)
        rows={}
        for Nx, Nt in RES:
            Phi,e=PHI_CACHE[(Nx,'Phi')],PHI_CACHE[(Nx,'e')]
            aa,ss=afun(rng)
            Ftr,U0,NU=true_field(Nx,Nt,seeds,aa,ss,xit,Phi,e)
            for m in METHODS:
                if m in kind: em,_,_=SC.eval_spectral(nets[m],kind[m],Nx,U0,NU,aa,ss,Ftr,PT[Nx])
                else: em,_,_=SC.eval_spatial(nets[m],Nx,U0,NU,aa,ss,Ftr)
                rows[(Nx,m)]=em
            log('  [%s %d^2] '%(tag,Nx+1)+' '.join('%s=%.2f%%'%(m,rows[(Nx,m)]) for m in METHODS))
        return rows
    log('\n===== B4 integer-order (alpha,s)=(1,1), classical variable-coefficient viscous Burgers =====')
    rInt = run_group('integer point', lambda r:(np.ones(NTE,np.float32), np.ones(NTE,np.float32)))
    log('\n===== B4 near-integer continuous neighborhood alpha,s~U[.9,1] =====')
    rNear = run_group('near integer', lambda r:(r.uniform(.9,1,NTE).astype(np.float32), r.uniform(.9,1,NTE).astype(np.float32)))
    log('\n===== B4 summary relL2(%) =====')
    for tag,R in [('integer (1,1)',rInt),('near-integer [.9,1]',rNear)]:
        log('--'+tag); log('resolution  '+''.join('%10s'%m for m in METHODS))
        for Nx,_ in RES:
            log('%-7s '%('%d^2'%(Nx+1))+''.join('%9.3f%%'%R[(Nx,m)] for m in METHODS))
    log('training time'+''.join('%9.1fs'%tt[m] for m in METHODS))
    np.savez(os.path.join(ROOT, 'b4_metrics.npz'),
             rint=np.array([[rInt[(Nx,m)] for m in METHODS] for Nx,_ in RES]),
             rnear=np.array([[rNear[(Nx,m)] for m in METHODS] for Nx,_ in RES]),
             methods=np.array(METHODS), res=np.array([Nx+1 for Nx,_ in RES]))
    open(os.path.join(ROOT, 'b4_result.txt'),'w',encoding='utf-8').write('\n'.join(L))

if __name__ == '__main__':
    main()
