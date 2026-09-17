# -*- coding: utf-8 -*-
"""
Ordered driver for the FrFNO reproduction package.

Each experiment script is self-contained and writes its result text file(s) to
the package root (the section folders hold the code and a copy of the expected
output for reference). Run everything in order with

    python run_all.py

or a single stage with

    python run_all.py --only sec5_2
    python run_all.py --list

Set FAST=1 for a quick smoke run (tiny fields / training steps; NOT the paper
numbers). The full run reproduces every table and figure in the manuscript on a
single RTX 3090; see README.md for the per-stage wall-clock budget.
"""
import argparse, os, subprocess, sys, time

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (key, description, [script paths relative to package root], run-from)
STEPS = [
    ('sec5_2', 'B1 main benchmark: FrFNO/PDNO/DeepONet/UNet + FNO/PINO/CNO', [
        r'sec5_2_B1_main\rerun_field_dim2.py',
        r'sec5_2_B1_main\rerun_3models.py',
        r'b1_burgers.py',                      # also writes b1_weights.pt for sec5_6
    ]),
    ('sec5_3', 'B2 long integration window (T=0.06)', [
        r'b2b_longT.py',                      # FrFNO/PDNO/DeepONet/UNet + b2b_weights.pt
        r'sec5_3_B2_longwindow\b2_frfno_regression.py',
    ]),
    ('sec5_4', 'B3 space-time fractional (PINO lambda sweep)', [
        r'sec5_4_B3_spacetime\b3_pino.py',
    ]),
    ('sec5_5', 'B4 integer-order limit', [
        r'b4_integer.py',
    ]),
    ('sec5_6', 'Surrogate speed-up / break-even (needs b1_weights.pt)', [
        r'sec5_6_speedup\speedup_benchmark.py',
        r'sec5_6_speedup\plot_speedup.py',
    ]),
    ('sec5_8', 'Component ablations C1 (B1) and C2 (B2)', [
        r'sec5_8_ablation\c_ablation.py',
        r'sec5_8_ablation\b2_ablation.py',
    ]),
    ('sec5_7', 'Theory verification Exp.1-2 and theory closure Exp.3-5', [
        r'sec5_7_theory\theory_verify.py',
        r'sec5_7_theory\theory_closure\p0_platform.py',
        r'sec5_7_theory\theory_closure\both_train.py',
        r'sec5_7_theory\theory_closure\p0_both_eval.py',
        r'sec5_7_theory\theory_closure\kscan_train.py',
        r'sec5_7_theory\theory_closure\trscan_train.py',
        r'sec5_7_theory\theory_closure\p0_trscan.py',
        r'sec5_7_theory\theory_closure\theory_p15_p14.py',
        r'sec5_7_theory\theory_closure\probe_spec.py',
    ]),
    ('sec5_9', 'Breadth: Riesz, reaction-diffusion, systems, NS, 3D', [
        r'sec5_9_breadth\riesz_periodic_check.py',
        r'sec5_9_breadth\reaction_diffusion_check.py',
        r'sec5_9_breadth\system_burgers_check.py',
        r'sec5_9_breadth\system_coupled_check.py',
        r'sec5_9_breadth\ns_vorticity_check.py',
        r'sec5_9_breadth\ns3d_vorticity_check.py',
    ]),
    ('appendixA', 'Appendix A: MMS convergence order verification', [
        r'appendixA_code_verification\mms_convergence.py',
    ]),
    ('figures', 'Analytic modal-decay schematic figure', [
        r'figures\make_modal_decay_fig.py',
    ]),
]


def run_script(rel, fast):
    path = os.path.join(PKG_ROOT, rel)
    env = dict(os.environ)
    if fast:
        env['FAST'] = '1'
    print('\n>>> python', rel, flush=True)
    t0 = time.time()
    r = subprocess.run([PY, path], cwd=PKG_ROOT, env=env)
    dt = time.time() - t0
    status = 'OK' if r.returncode == 0 else f'FAIL(rc={r.returncode})'
    print(f'<<< [{status}] {rel}  ({dt/60:.1f} min)', flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='run a single stage key, e.g. sec5_2')
    ap.add_argument('--fast', action='store_true', help='FAST=1 smoke run')
    ap.add_argument('--list', action='store_true', help='list stages and exit')
    args = ap.parse_args()

    if args.list:
        for k, desc, scr in STEPS:
            print(f'{k:9s} {desc}  [{len(scr)} script(s)]')
        return

    steps = [s for s in STEPS if (args.only is None or s[0] == args.only)]
    if args.only and not steps:
        raise SystemExit(f'unknown stage {args.only}; use --list')

    failed = []
    t_all = time.time()
    for k, desc, scripts in steps:
        print('\n' + '=' * 78 + f'\nSTAGE {k}: {desc}\n' + '=' * 78, flush=True)
        for rel in scripts:
            if not run_script(rel, args.fast):
                failed.append(rel)
    print('\n' + '#' * 78)
    print(f'all stages finished in {(time.time()-t_all)/3600:.2f} h')
    if failed:
        print('FAILED SCRIPTS:'); [print('  -', x) for x in failed]
        raise SystemExit(1)
    print('no failures.')


if __name__ == '__main__':
    main()
