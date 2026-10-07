# -*- coding: utf-8 -*-
"""
Fig: lowest-mode Mittag-Leffler decay E_alpha(-c T^alpha lambda_k^s) vs T.
Left : mode (1,1), several (alpha,s), mark T=0.015 (main) and T=0.06 (B2).
Right: fixed (alpha,s)=(.75,.75), modes (1,1),(4,4),(8,8).
Numbers are the exact series for small z and the standard asymptotic tail for
large z (where the power series suffers catastrophic cancellation), blended.
"""
import os
import numpy as np
from scipy.special import gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                     'axes.linewidth': .8, 'lines.linewidth': 1.6,
                     'mathtext.fontset': 'cm'})

c = 1.0
LAM = {(1, 1): 2*np.pi**2, (4, 4): 32*np.pi**2, (8, 8): 128*np.pi**2}

def ml_series(a, z, tol=1e-13, mmax=400):
    """E_a(-z) by power series (safe for small/moderate z)."""
    z = np.asarray(z, float); out = np.zeros_like(z)
    for i, zz in enumerate(z.flat):
        s, term = 1.0, 1.0
        for m in range(1, mmax + 1):
            term = term * (-zz) * gamma(1 + a*(m-1)) / gamma(1 + a*m)
            s += term
            if abs(term) < tol * max(1.0, abs(s)):
                break
        out.flat[i] = s
    return out.reshape(z.shape)

def Ealpha(a, z):
    z = np.asarray(z, float)
    if abs(a - 1.0) < 1e-12:
        return np.exp(-z)
    ser = ml_series(a, np.clip(z, None, 6.0))
    asy = 1.0 / (gamma(1 - a) * np.maximum(z, 1e-12))     # first-order asymp tail
    # series for z<=5, asymp for z>=7, linear blend in between
    w = np.clip((z - 5.0) / 2.0, 0.0, 1.0)
    return np.where(z <= 5.0, ser, np.where(z >= 7.0, asy, (1 - w)*ser + w*asy))

def decay(a, s, lam, T):
    return Ealpha(a, c * T**a * lam**s)

T = np.linspace(1e-3, 0.10, 600)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 2.95))

# ---- left: lowest mode, several orders ----
lam11 = LAM[(1, 1)]
orders = [(1.0, 1.0), (0.9, 0.9), (0.75, 0.75), (0.6, 0.6), (0.75, 0.5)]
styles = {('$(\u03b1,s)=(1.00,1.00)$',): None}
labels = [r'$(\alpha,s)=(1.00,1.00)$', r'$(0.90,0.90)$',
          r'$(0.75,0.75)$ (main)', r'$(0.60,0.60)$', r'$(0.75,0.50)$']
cols = ['#888888', '#4C78A8', '#D1495B', '#2A9D8F', '#E69F00']
lws = [1.4, 1.4, 2.4, 1.4, 1.4]
ls = ['--', '-', '-', '-', ':']
for (a, s), lab, col, lw, l in zip(orders, labels, cols, lws, ls):
    axL.plot(T, decay(a, s, lam11, T), l, color=col, lw=lw, label=lab)
for Tv, txt in [(0.015, r'$T=0.015$'), (0.06, r'$T=0.06$')]:
    axL.axvline(Tv, color='k', ls=':', lw=.9, alpha=.55)
    if Tv > .03:
        axL.text(Tv - 0.006, 0.90, txt, va='top', ha='right', fontsize=7.2,
                 bbox=dict(boxstyle='round,pad=.15', fc='white', ec='none', alpha=.85))
    else:
        axL.text(Tv + 0.0016, 0.965, txt, va='top', ha='left', fontsize=7.2,
                 bbox=dict(boxstyle='round,pad=.15', fc='white', ec='none', alpha=.85))
# key points on the main curve
for Tv in [0.015, 0.06]:
    yv = float(decay(0.75, 0.75, lam11, np.array([Tv]))[0])
    axL.plot([Tv], [yv], 'o', ms=4.2, color='#D1495B', zorder=5)
    axL.annotate(f'{yv:.3f}', (Tv, yv), textcoords='offset points',
                 xytext=(7, 5), fontsize=7.6, color='#D1495B')
axL.set_xlabel(r'terminal time $T$'); axL.set_ylabel(r'lowest-mode amplitude  $E_\alpha(-cT^\alpha\lambda_{(1,1)}^s)$')
axL.set_xlim(0, .10); axL.set_ylim(0, 1.02)
axL.legend(frameon=False, fontsize=7.3, loc='upper right')
axL.set_title('(a) lowest mode, varying fractional order', fontsize=8.6)

# ---- right: fixed order, several spatial modes ----
mode_lab = {('(1,1)',): None}
for key, col, mk in [((1, 1), '#D1495B', 'o'), ((4, 4), '#4C78A8', 's'), ((8, 8), '#2A9D8F', '^')]:
    axR.plot(T, decay(0.75, 0.75, LAM[key], T), color=col, lw=1.8,
             label=r'mode $\mathbf{k}=(%d,%d)$' % key)
axR.axvline(0.015, color='k', ls=':', lw=.9, alpha=.55)
axR.text(0.0166, 0.965, r'$T=0.015$', va='top', ha='left', fontsize=7.2,
         bbox=dict(boxstyle='round,pad=.15', fc='white', ec='none', alpha=.85))
for key, col in [((1, 1), '#D1495B'), ((4, 4), '#4C78A8'), ((8, 8), '#2A9D8F')]:
    yv = float(decay(0.75, 0.75, LAM[key], np.array([0.015]))[0])
    axR.plot([0.015], [yv], 'o', ms=3.6, color=col, zorder=5)
    axR.annotate(f'{yv:.3f}', (0.015, yv), textcoords='offset points',
                 xytext=(5, -2), fontsize=7.2, color=col)
axR.set_xlabel(r'terminal time $T$')
axR.set_ylabel(r'mode amplitude  $E_{0.75}(-cT^{0.75}\lambda_{\mathbf{k}}^{0.75})$')
axR.set_xlim(0, .10); axR.set_ylim(0, 1.02)
axR.legend(frameon=False, fontsize=7.4, loc='upper right')
axR.set_title(r'(b) fixed $(\alpha,s)=(0.75,0.75)$, varying mode', fontsize=8.6)

for ax in (axL, axR):
    ax.tick_params(direction='in', length=3, width=.8)

fig.tight_layout()
_OUTDIR = os.path.dirname(os.path.abspath(__file__))
out_pdf = os.path.join(_OUTDIR, 'modal_decay_vs_T.pdf')
out_png = os.path.join(_OUTDIR, 'modal_decay_vs_T.png')
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, dpi=200, bbox_inches='tight')
print('saved', out_pdf)
# echo the exact key numbers used
for Tv in [0.015, 0.06]:
    print('T=%.3f (1,1)=%.4f' % (Tv, float(decay(.75, .75, LAM[(1,1)], np.array([Tv]))[0])))
for key in [(1,1),(4,4),(8,8)]:
    print('T=0.015 mode%s = %.4f' % (key, float(decay(.75,.75,LAM[key],np.array([.015]))[0])))
