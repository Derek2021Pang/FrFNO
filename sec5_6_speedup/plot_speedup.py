# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""Visualize the surrogate speedup benchmark results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.linewidth': 0.8,
                     'mathtext.fontset': 'cm'})

import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_out_dir = _os.path.join(_os.path.dirname(_PKG_ROOT), 'FrFNO-JSC', 'figs')
if not _os.path.isdir(_out_dir): _out_dir = _os.path.join(_PKG_ROOT, 'figures')
_os.makedirs(_out_dir, exist_ok=True)
out_pdf = _os.path.join(_out_dir, 'speedup_benchmark.pdf')
out_png = _os.path.join(_out_dir, 'speedup_benchmark.png')

# ---- measured data (from speedup_benchmark_result.txt, rerun 2026-10-06) ----
res = ['$17^2$', '$65^2$', '$129^2$']
nl_single = [2045.81, 8464.04, 16048.81]   # nonlinear solver B=1, ms
fr_single = [12.446, 14.439, 34.852]       # FrFNO B=1, ms
nl_batch  = [34.605, 132.963, 2086.717]    # nonlinear solver B=64 amortized, ms
fr_batch  = [12.7583, 13.3806, 14.0030]    # FrFNO B=64 amortized, ms
speedup_single = [164.4, 586.2, 460.5]

C_OFF = 281.6  # offline seconds
nq = np.array([1, 10, 50, 100, 500, 1000, 5000])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# ---- left: per-query wall-clock (log scale) ----
x = np.arange(3); w = 0.18
ax1.bar(x - 1.5*w, nl_single, w, label='Nonlinear solver, $B=1$', color='#c0392b')
ax1.bar(x - 0.5*w, fr_single, w, label='FrFNO, $B=1$', color='#2471a3')
ax1.bar(x + 0.5*w, nl_batch, w, label='Solver, $B=64$ amortized', color='#e59866')
ax1.bar(x + 1.5*w, fr_batch, w, label='FrFNO, $B=64$ amortized', color='#76d7c4')
ax1.set_yscale('log'); ax1.set_ylabel('Wall-clock per query (ms)')
ax1.set_xticks(x); ax1.set_xticklabels(res)
ax1.set_title('(a) Per-query cost')
ax1.legend(fontsize=8.5, loc='upper left')
ax1.set_ylim(5, 3e4)
for i, sp in enumerate(speedup_single):
    ax1.annotate('%.0fx' % sp, (x[i], fr_single[i]*0.55), ha='center',
                 fontsize=9, color='#1a5276', fontweight='bold')

# ---- right: total wall-clock vs number of queries (129^2) ----
colors = {'$17^2$': '#85929e', '$65^2$': '#dc7633', '$129^2$': '#922b21'}
markers = {'$17^2$': 'o', '$65^2$': 's', '$129^2$': '^'}
solver_s = {'$17^2$': 2.04581, '$65^2$': 8.46404, '$129^2$': 16.04881}
fr_s = {'$17^2$': 0.012446, '$65^2$': 0.014439, '$129^2$': 0.034852}
for r in res:
    ts = nq * solver_s[r]
    tf = C_OFF + nq * fr_s[r]
    ax2.loglog(nq, ts, '--' + markers[r], color=colors[r], ms=5, label='Solver %s' % r)
    ax2.loglog(nq, tf, '-' + markers[r], color=colors[r], ms=5, label='FrFNO %s' % r)
# break-even markers
be = {'$17^2$': 138, '$65^2$': 33, '$129^2$': 17}
for r in res:
    nb = be[r]; tt = C_OFF + nb*fr_s[r]
    ax2.plot(nb, tt, 'k*', ms=12, zorder=5)
ax2.axhline(C_OFF, color='gray', ls=':', lw=0.8)
ax2.text(1.1, C_OFF*1.15, 'offline cost %.0f s' % C_OFF, fontsize=8.5, color='gray')
ax2.set_xlabel('Number of queries'); ax2.set_ylabel('Total wall-clock (s)')
ax2.set_title('(b) Total cost vs query count ($\\star$ = break-even)')
ax2.legend(fontsize=7.5, ncol=2, loc='upper left')
ax2.grid(True, which='both', ls=':', alpha=0.4)

plt.tight_layout()
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print('saved', out_pdf)
