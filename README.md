# FrFNO: Fractional Fourier Neural Operator

Code availability: https://github.com/Derek2021Pang/FrFNO

FrFNO is a *conditional* neural operator for space–time fractional PDEs. The
exact modal response of the frozen linear part — the one-parameter
Mittag–Leffler / L1 multiplier — is precomputed **once** as a
**one-dimensional, resolution-independent propagator table**, and a spectral
convolutional network is trained only on the residual caused by variable
coefficients and nonlinear advection, conditioned continuously on both fractional
orders $(\alpha,s)$. It supports zero-shot spatial super-resolution.

All experiments use one GPU (single NVIDIA RTX 3090, 24 GB), PyTorch 2.4.1
(CUDA 12.4), float32. The numbers in the paper correspond to the
2026-10-06 rerun; see `RUNLOG_20261006.md` for the run status of every
experiment.

---

## 1. Repository layout

Code is organised **by experiment**. The shared library lives at the package
root (so every driver can import it); each `secX_*` folder contains the driver
script(s) that produce that experiment's numbers plus a copy of the expected
text output for reference.

```
FrFNO_repro/
├── frfno_core.py          # propagator table (1-D), FrFNO/FNO DualNet, analytic base, conditioning
├── d_scan.py              # KL basis, random log-normal diffusivity, order grids
├── nl_solver.py           # nonlinear reference solver (GPU, upwind advection, implicit L1)
├── gpu_solver.py          # batched spectral/DST GPU primitives and IMEX solver
├── models_surrogate.py    # FNO/PINO wrapper, CNOWrap, DeepONet2d, UNet2d, parameter counts
├── models_extra.py        # PDNO pseudo-differential operator baseline
├── surrogate_compare.py   # shared train/eval harness used by ablations and breadth tests
├── b1_burgers.py          # main benchmark driver (all 7 models, T=0.015)
├── b2b_longT.py           # long-window driver (all 7 models, T=0.06)
├── b4_integer.py          # integer-order driver (all 7 models, (1,1) + near-integer)
├── fractional_pde_study/  # spectral reference solvers (solver*.py)
├── CNO2d_simplified/      # vendored CNO block (CNO2d.py)
│
├── sec5_2_B1_main/        # main benchmark (2-D space-time fractional Burgers)
├── sec5_3_B2_longwindow/  # long integration window (T=0.06)
├── sec5_4_B3_spacetime/   # space-time variant + PINO residual-weight sweep
├── sec5_5_B4_integer/     # integer-order limit
├── sec5_6_speedup/        # per-query cost, break-even, total wall-clock
├── sec5_7_theory/         # theory verification Exp.1-2; theory_closure/ = Exp.3-5
├── sec5_8_ablation/       # component ablations
├── sec5_9_breadth/        # Riesz, reaction-diffusion, systems, fractional NS, 3-D
├── appendixA_code_verification/ # Method-of-Manufactured-Solutions convergence orders
├── figures/               # analytic modal-decay schematic
├── run_all.py             # ordered end-to-end driver
└── requirements.txt
```

### Portability convention

A driver lives one level below the root, so each starts with a small bootstrap
that puts the package root on `sys.path`; it then writes result files to the
**package root** (the driver folder keeps a reference copy). There are no
personal absolute paths and the package runs from any location. The
`theory_closure/` scripts sit two levels below the root and additionally expose
their own folder and the `sec5_7_theory` folder for intra-package imports.
The `theory_closure/` folder at the package root stores the Exp.3/4 text
outputs (`p15_p14_*.txt`) and the Exp.5 weight/`npz` artefacts; the driver
scripts live in `sec5_7_theory/theory_closure/`.

---

## 2. Environment setup

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
# install a CUDA-enabled torch matching your driver, e.g. CUDA 12.4:
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Smoke test (tiny budget, seconds-to-minutes; NOT the full numbers):

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
| Backbone size | ≈1880 K parameters for FrFNO/FNO/PINO (matched comparisons) |
| CNO policy | official recipe: `use_bn=False`, AdamW(1e-3, wd 1e-8), L1 loss |
| FNO/PINO inputs | `field_dim=2` (fractional low-pass features on **both** x and y coordinate fields) |

The propagator table is the **one-dimensional** table
`build_prop_table_1d(na=41, ns=41, Nt=400, M=1600)` on the coordinate
$z=\lambda^s$: because the frozen scalar multiplier depends on $(\alpha,z)$ but
not on an independent $s$ grid, no $s$-interpolation is performed.

---

## 4. Experiment → script → output file map

The authoritative output text files are written to the package root (Exp.5
writes into `theory_closure/`). A fresh run overwrites the same file names.
Run status (last rerun date) is tracked in `RUNLOG_20261006.md`.

### Main benchmarks

| Experiment | Driver script(s) | Output file | Rerun |
|---|---|---|---|
| Main benchmark B1, all 7 models | `b1_burgers.py` (root) | `b1_burgers_result.txt` | 2026-10-06 |
| rollout mechanism diagnostic (supporting diagnostics for the one-shot design; not part of the paper) | `rollout_diagnose.py` (root) | `rollout_diagnose_result.txt` | 2026-10-07 |
| semigroup test (supporting diagnostics; linear-table $E(2\tau)$ vs $E(\tau)E(\tau)$; not part of the paper) | `semigroup_test.py` (root) | `semigroup_test_result.txt` | 2026-10-07 |
| table-shift test (supporting diagnostics; two-step 400-step table, first- vs second-half kernel; not part of the paper) | `table_shift_test.py` (root) | `table_shift_test_result.txt` | 2026-10-07 |
| Long-window B2, all 7 models | `b2b_longT.py` (root) | `b2b_longT_result.txt` | 2026-10-06 |
| Integer-order B4, all 7 models | `b4_integer.py` (root) | `b4_result.txt` | 2026-10-06 |
| Space-time variant (PINO λ sweep) | `sec5_4_B3_spacetime/b3_pino.py` | `b3_result_lam0.txt`, `b3_result_lam0.5.txt` | 2026-10-07 |

The 1-D/2-D table regression diagnostics (`sec5_2_B1_main/b1_frfno_regression.py`,
`sec5_3_B2_longwindow/b2_frfno_regression.py`) are historical checks and are not
a paper data source.

### Speed-up

| Experiment | Driver | Output | Rerun |
|---|---|---|---|
| per-query cost / break-even / total-cost | `sec5_6_speedup/speedup_benchmark.py` | `speedup_benchmark_result.txt` | 2026-10-06 |
| total wall-clock vs number of queries plot | `sec5_6_speedup/plot_speedup.py` | `../FrFNO-JSC/figs/speedup_benchmark.pdf/.png` | 2026-10-07 |

`speedup_benchmark.py` loads `b1_weights.pt`, which is produced by running
`b1_burgers.py` (stage `sec5_2` in `run_all.py` does this first).

### Ablations

| Experiment | Driver | Output | Rerun |
|---|---|---|---|
| Component ablation (full / no_base / no_dyn / no_nb, seeds, data, width) | `sec5_8_ablation/c_ablation.py` | `c_ablation_result.txt` | 2026-10-07 |
| Long-window ablation | `sec5_8_ablation/b2_ablation.py` | `b2_ablation_result.txt` | 2026-10-06 |

### Theory verification

| Experiment | Driver | Output | Rerun |
|---|---|---|---|
| Exp.1 grid increment, Exp.2 s-scaling FrFNO vs FNO | `sec5_7_theory/theory_verify.py` | `theory_verify_result.txt` | 2026-10-06 |
| Exp.3/4 slopes (T, η, β, Born ratio) | `sec5_7_theory/theory_closure/theory_p15_p14.py` | `theory_closure/p15_p14_{T,eta,beta,spec,summary}.txt` | 2026-10-06 |
| Exp.5 error-floor / two-bandwidth (K-scan, train-resolution scan, spectral tail, K-bottleneck) | `sec5_7_theory/theory_closure/p0_platform.py`, `both_train.py`, `p0_both_eval.py`, `kscan_train.py`, `trscan_train.py`, `p0_trscan.py`, `exp5_supplement.py`, `exp5_validate.py` | `theory_closure/p0_platform.log`, `p0_both_eval.log`, `p0_trscan.log`, `exp5_supp.log`, `exp5_validate.log`, `kscan_rerun_20261006.log`, `trscan*_rerun_20261006.log`, `both*_rerun_20261006.log`, `kscan_N64_rerun_20261006.log`, `p0_*.npz`, `exp5_*.npz` | 2026-10-06 |

### Breadth (other equation classes)

| Equation | Driver | Output | Rerun |
|---|---|---|---|
| Periodic Riesz fractional Laplacian | `sec5_9_breadth/riesz_periodic_check.py` | `riesz_periodic_result.txt` | 2026-10-06 |
| Linear fractional diffusion / Fisher-KPP / Allen–Cahn | `sec5_9_breadth/reaction_diffusion_check.py` | `reaction_diffusion_result.txt` | 2026-10-06 |
| 2-D two-component vector Burgers | `sec5_9_breadth/system_burgers_check.py` | `system_burgers_result.txt` | 2026-10-06 |
| Coupled system with non-symmetric J | `sec5_9_breadth/system_coupled_check.py` | `system_coupled_result.txt` | 2026-10-06 |
| 2-D fractional NS (vorticity) | `sec5_9_breadth/ns_vorticity_check.py` | `ns_vorticity_result.txt` | 2026-10-06 |
| 3-D fractional NS (vector vorticity) | `sec5_9_breadth/ns3d_vorticity_check.py` | `ns3d_result.txt` | 2026-10-06 |

### Reference-solver verification and plots

| Item | Driver | Output | Rerun |
|---|---|---|---|
| MMS temporal/spatial convergence orders (+ t^α weak-singularity test) | `appendixA_code_verification/mms_convergence.py` | `mms_result.txt` | 10-05 |
| Modal-decay schematic (analytic) | `figures/make_modal_decay_fig.py` | `figs/modal_decay_vs_T.pdf/.png` | — |
| Solution snapshot / band-filling phase error | `sec5_2_B1_main/plot_structure_b1.py`, `plot_band_filling.py` | `../FrFNO-JSC/figs/fig_snapshot_structure.png`, `fig_band_filling.png`, `phase_ncc_result.txt` | 2026-10-07 |
| NumPy vs GPU IMEX cross-validation (float64) | `numpy_verify_solver.py` (root) | `numpy_verify_result.txt` | 2026-10-07 |

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

Every large binary is committed and hosted on GitHub. To respect the GitHub
100 MB per-file limit, the large files use two mechanisms:

- **`data_arrays.zip` (Git LFS, 297 MB)**: bundles all 48 loose `.npz` arrays
  (534 MB uncompressed, deflate level 9), including
  `theory_closure/p0_ref513.npz` (the 513² frozen ground-truth fields for the
  Exp.5 error-floor tiers) and `theory_closure/trainset_N64.npz` (the 64²
  training set for the K-scan N64 models). After cloning, unzip it at the
  repository root to restore every loose `.npz` path that the drivers expect:
  ```bash
  unzip data_arrays.zip        # Windows: Expand-Archive data_arrays.zip -DestinationPath .
  ```
- **`.pt` weights above 100 MB (Git LFS)**: `c_weights.pt` (256 MB),
  `theory_closure/both_K24_Nx64.pt` (162 MB), `theory_closure/kscan_K24.pt`
  (162 MB). Git LFS downloads them automatically during `git clone`.

All other `.pt` weights (B1/B2b/B3/B4, 57-73 MB) are committed directly.
Everything is regenerable by `python run_all.py`, so the LFS files are
convenience artefacts rather than requirements. The small `*_result.txt` /
`*.out` / summary files shipped in each driver folder are the exact outputs
obtained on the reference machine and are provided for diffing.

## 6. Notes

* The headline numbers use `frfno_core.py` with the one-dimensional propagator
  table. There is no `frfno_legacy.py` in this repository.
* The 1-D propagator table is built deterministically and cached to
  `prop_table_1d_na*_ns*_Nt*_M*_T*.npz` in the package root on first use; a
  fresh process loads it instead of recomputing (10.3 MB for the default
  `(41,41,400,1600)`). The file is git-ignored and regenerated automatically if
  missing or corrupt.
* `RUNLOG_20261006.md` records the last rerun date of every experiment; all
  paper numbers now sit on the 2026-10-06/07 rerun basis (MMS: 10-05, confirmed
  unaffected).
* Set `FAST=1` only for pipeline smoke tests; it shrinks fields and steps and
  does not reproduce the full numbers.
