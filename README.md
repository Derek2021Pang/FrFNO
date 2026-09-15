# FrFNO: Fractional Fourier Neural Operator

Reproduction code for **"FrFNO: Injecting the analytic Mittag--Leffler propagator
into a discretization-convergent neural operator for space--time fractional
PDEs"** (submitted to Journal of Computational Physics).

Code availability: https://github.com/guofeipang/FrFNO

FrFNO is a *conditional* neural operator for space–time fractional PDEs. The
exact modal response of the frozen linear part — the one-parameter
Mittag–Leffler / L1 multiplier — is precomputed **once** as a
**one-dimensional, resolution-independent propagator table**, and a spectral
convolutional network is trained only on the residual caused by variable
coefficients and nonlinear advection, conditioned continuously on both fractional
orders $(\alpha,s)$. It supports zero-shot spatial super-resolution and ships
with a matching discretization-error analysis.

All experiments use one GPU (single NVIDIA RTX 3090, 24 GB), PyTorch 2.4.1
(CUDA 12.4), float32.

---

## 1. Repository layout

Code is organised **by paper section**. The shared library lives at the package
root (so every section driver can import it); each `secX_*` folder contains the
driver script(s) that produce that section's numbers plus a copy of the expected
text output for reference.

```
FrFNO/
├── frfno_core.py          # propagator table (1-D), FrFNO/FNO DualNet, analytic base, conditioning
├── d_scan.py              # KL basis, random log-normal diffusivity, order grids
├── nl_solver.py           # nonlinear reference solver (GPU, upwind advection, implicit L1)
├── gpu_solver.py          # batched spectral/DST GPU primitives and IMEX solver
├── models_surrogate.py    # FNO/PINO wrapper, CNOWrap, DeepONet2d, UNet2d, parameter counts
├── models_extra.py        # PDNO pseudo-differential operator baseline
├── surrogate_compare.py   # shared train/eval harness used by ablations and breadth tests
├── b1_burgers.py          # B1 data/ground-truth module (also a runnable driver; writes b1_weights.pt)
├── b2b_longT.py           # B2 data module + driver (writes b2b_weights.pt)
├── b4_integer.py          # B4 data module + driver (writes b4_weights.pt)
├── frfno_legacy.py        # legacy module used ONLY by sec5_7/theory_closure (Exp.3-5)
├── fractional_pde_study/  # spectral reference solvers (solver*.py)
├── CNO2d_simplified/      # vendored CNO block (CNO2d.py)
│
├── sec5_2_B1_main/        # main benchmark B1 (2-D space-time fractional Burgers)
├── sec5_3_B2_longwindow/  # long integration window B2 (T=0.06)
├── sec5_4_B3_spacetime/   # space-time fractional B3 + PINO residual-weight sweep
├── sec5_5_B4_integer/     # integer-order limit B4 (driver b4_integer.py at root)
├── sec5_6_speedup/        # per-query cost, break-even, total wall-clock figure
├── sec5_7_theory/         # theory verification Exp.1-2; theory_closure/ = Exp.3-5
├── sec5_8_ablation/       # component ablations C1 (B1) and C2 (B2)
├── sec5_9_breadth/        # Riesz, reaction-diffusion, systems, fractional NS, 3-D
├── appendixA_code_verification/ # Method-of-Manufactured-Solutions convergence orders
├── figures/               # analytic modal-decay schematic
├── run_all.py             # ordered end-to-end driver
└── requirements.txt
```

### Portability convention

A section driver lives one level below the root, so each starts with a small
bootstrap that puts the package root on `sys.path`; it then writes result files
to the **package root** (the section folder keeps a reference copy). There are no
personal absolute paths and the package runs from any location. The
`theory_closure/` scripts sit two levels below the root and additionally expose
their own folder and the `sec5_7_theory` folder for intra-package imports.

---

## 2. Environment setup

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
# install a CUDA-enabled torch matching your driver, e.g. CUDA 12.4:
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Smoke test (tiny budget, seconds-to-minutes; NOT the paper numbers):

```bash
set FAST=1 & python run_all.py --only sec5_2
```

---

## 3. Common experimental protocol (identical for every method)

| Item | Value |
|---|---|
| Training grid | internal 15² (Nx=16), i.e. 17² points including boundaries |
| Training fields / orders | 128 random fields × 9×9 order grid (10 368 instances) |
| Optimiser / schedule | Adam 1.5e-3, weight decay 1e-5, cosine schedule, 8000 steps |
| Test orders | 48 held-out $(\alpha,s)$ pairs, `RandomState(2024)` |
| Evaluation tiers | train 17² and zero-shot super-resolution 65², 129² |
| Error metric | relative $L^2$ error (%) against the same GPU reference solver |
| Backbone size | ≈1.88M parameters for FrFNO/FNO/PINO (matched single-knockout comparisons) |
| CNO policy | official recipe: `use_bn=False`, AdamW(1e-3, wd 1e-8), L1 loss |
| FNO/PINO inputs | `field_dim=2` (fractional low-pass features on **both** x and y coordinate fields) |

The propagator table is the **one-dimensional** table
`build_prop_table_1d(na=41, ns=41, Nt=400, M=1600)` on the coordinate
$z=\lambda^s$: because the frozen scalar multiplier depends on $(\alpha,z)$ but
not on an independent $s$ grid, no $s$-interpolation is performed. All FrFNO
numbers in the paper use this table.

---

## 4. Paper result → script → output file map

> A reported table is sometimes the concatenation of two scripts (FrFNO-family
> columns and the re-run competitor columns). The authoritative output text file
> is listed for each. Reference copies sit next to the driver; a fresh run writes
> the same file name to the package root.

### Main benchmarks

| Paper item | Driver script(s) | Output file |
|---|---|---|
| Table B1 — FrFNO, PDNO, DeepONet, U-Net | `sec5_2_B1_main/rerun_field_dim2.py` | `rerun_field_dim2.log` |
| Table B1 — FNO, PINO, CNO | `sec5_2_B1_main/rerun_3models.py` (B1 block) | `rerun_3models_result.txt` |
| Table B2 — FrFNO, PDNO, DeepONet, U-Net | `b2b_longT.py` (root) | `b2b_longT_result.txt` |
| Table B2 — FNO, PINO, CNO | `sec5_2_B1_main/rerun_3models.py` (B2 block) | `rerun_3models_result.txt` |
| Table B3 (PINO λ sweep) | `sec5_4_B3_spacetime/b3_pino.py` | `b3_result_lam0.5.txt`, `b3_result_lam0.txt` |
| Table B4 (integer order) | `b4_integer.py` (root) | `b4_result.txt` |
| 1-D vs 2-D table regression check | `sec5_2_B1_main/b1_frfno_regression.py`, `sec5_3_B2_longwindow/b2_frfno_regression.py` | `b1/b2_frfno_regression_result.txt` |

### Speed-up (Section 5.6)

| Paper item | Driver | Output |
|---|---|---|
| per-query cost / break-even / total-cost table | `sec5_6_speedup/speedup_benchmark.py` | `speedup_benchmark_result.txt` |
| total wall-clock vs number of queries figure | `sec5_6_speedup/plot_speedup.py` | `speedup_benchmark.pdf/.png` |

`speedup_benchmark.py` loads `b1_weights.pt`, which is produced by running
`b1_burgers.py` (stage `sec5_2` in `run_all.py` does this first).

### Ablations (Section 5.8)

| Paper item | Driver | Output |
|---|---|---|
| Table C1 (full / no_base / no_dyn / no_nb, seeds, data, width) | `sec5_8_ablation/c_ablation.py` | `c_ablation_result.txt` |
| Table C2 (B2 long-window ablation) | `sec5_8_ablation/b2_ablation.py` | `b2_ablation_result.txt` |

### Theory verification (Section 5.7 and Appendix)

| Paper item | Driver | Output |
|---|---|---|
| Exp.1 grid increment, Exp.2 s-scaling FrFNO vs FNO | `sec5_7_theory/theory_verify.py` | `theory_verify_result.txt` |
| Exp.3/4 slopes (T, η, β, Born ratio) | `sec5_7_theory/theory_closure/theory_p15_p14.py` | `p15_p14_{T,eta,beta,spec,summary}.txt` |
| Exp.5 error-floor / two-bandwidth (K-scan, train-resolution scan) | `theory_closure/p0_platform.py`, `both_train.py`, `p0_both_eval.py`, `kscan_train.py`, `trscan_train.py`, `p0_trscan.py` | `p0eval.out`, `p0treval.out`, `exp5_both_summary.md` |

### Breadth (Section 5.9)

| Equation | Driver | Output |
|---|---|---|
| Periodic Riesz fractional Laplacian | `sec5_9_breadth/riesz_periodic_check.py` | `riesz_periodic_result.txt` |
| Linear fractional diffusion / Fisher-KPP / Allen–Cahn | `sec5_9_breadth/reaction_diffusion_check.py` | `reaction_diffusion_result.txt` |
| 2-D two-component vector Burgers | `sec5_9_breadth/system_burgers_check.py` | `system_burgers_result.txt` |
| Coupled system with non-symmetric J | `sec5_9_breadth/system_coupled_check.py` | `system_coupled_result.txt` |
| 2-D fractional NS (vorticity) | `sec5_9_breadth/ns_vorticity_check.py` | `ns_vorticity_result.txt` |
| 3-D fractional NS (vector vorticity) | `sec5_9_breadth/ns3d_vorticity_check.py` | `ns3d_result.txt` |

### Appendix and figures

| Paper item | Driver | Output |
|---|---|---|
| MMS temporal/spatial convergence orders | `appendixA_code_verification/mms_convergence.py` | `mms_result.txt` |
| Modal-decay schematic (analytic) | `figures/make_modal_decay_fig.py` | `figures/modal_decay_vs_T.pdf/.png` |
| B1 solution snapshot / radial spectrum | `sec5_2_B1_main/plot_structure_b1.py`, `plot_spectrum_b1.py` | `fig_snapshot_structure.png`, `fig_radial_spectrum.png` |
| Algorithms 1–3 | implemented in `frfno_core.py`, `models_surrogate.py` | — |

---

## 5. End-to-end reproduction

```bash
python run_all.py --list      # show ordered stages
python run_all.py             # full reproduction in dependency order
python run_all.py --only sec5_6   # one stage
```

Stages are ordered so that weights needed downstream exist
(`b1_weights.pt`, `b2b_weights.pt`, `b4_weights.pt` are created by the training
stages and consumed by the speed-up / plotting stages). Long scripts checkpoint
incrementally (`*.npz`) and stream results to their text file, so an interrupted
run can be resumed by re-launching the same stage.

Trained weights and large reference arrays (`*.pt`, multi-MB `*.npz`) are **not**
committed; they are regenerated by the training stages. The small `*_result.txt`
/ `*.out` / summary files shipped in each section folder are the exact outputs
obtained on the reference machine and are provided for diffing.

## 6. Notes

* `frfno_legacy.py` is the renamed predecessor module used only by the
  theory-closure experiments; the headline results all use `frfno_core.py`.
* Set `FAST=1` only for pipeline smoke tests; it shrinks fields and steps and
  does not reproduce the manuscript.
