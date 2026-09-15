# Code ↔ numerical-result map (FrFNO paper)

Every table/figure/algorithm in the manuscript, the script that generates it, and
the authoritative output file. Paths are relative to the package root. Section
folders also carry a reference copy of each output text file. All FrFNO numbers
use the **one-dimensional** propagator table (`frfno_core.build_prop_table_1d`).

| # | Paper location | Kind | Generating script | Output file / artifact |
|---|---|---|---|---|
| 1 | Table B1 (Sec 5.2) | FrFNO, PDNO, DeepONet, U-Net | `sec5_2_B1_main/rerun_field_dim2.py` | `rerun_field_dim2.log` |
| 2 | Table B1 (Sec 5.2) | FNO, PINO, CNO | `sec5_2_B1_main/rerun_3models.py` (B1 block) | `rerun_3models_result.txt` |
| 3 | Table B2 (Sec 5.3) | FrFNO, PDNO, DeepONet, U-Net | `b2b_longT.py` | `b2b_longT_result.txt` |
| 4 | Table B2 (Sec 5.3) | FNO, PINO, CNO | `sec5_2_B1_main/rerun_3models.py` (B2 block) | `rerun_3models_result.txt` |
| 5 | Table B3 (Sec 5.4) | FrFNO / FNO / PINO λ-sweep | `sec5_4_B3_spacetime/b3_pino.py` | `b3_result_lam0.txt`, `b3_result_lam0.5.txt` |
| 6 | Table B4 (Sec 5.5) | integer-order, all methods | `b4_integer.py` (+ `rerun_3models.py` competitor block) | `b4_result.txt` |
| 7 | Table per-query cost (Sec 5.6) | direct vs FrFNO timing | `sec5_6_speedup/speedup_benchmark.py` | `speedup_benchmark_result.txt` |
| 8 | Table break-even / offline (Sec 5.6) | cost analysis | `sec5_6_speedup/speedup_benchmark.py` | `speedup_benchmark_result.txt` |
| 9 | Fig. total wall-clock (Sec 5.6) | figure | `sec5_6_speedup/plot_speedup.py` | `speedup_benchmark.pdf/.png` |
| 10 | Table C1 (Sec 5.8) | B1 component ablation | `sec5_8_ablation/c_ablation.py` | `c_ablation_result.txt` |
| 11 | Table C2 (Sec 5.8) | B2 component ablation | `sec5_8_ablation/b2_ablation.py` | `b2_ablation_result.txt` |
| 12 | Exp.1–2 (Sec 5.7) | theory verification | `sec5_7_theory/theory_verify.py` | `theory_verify_result.txt` |
| 13 | Exp.3–4 (Sec 5.7/App.) | slopes T, η, β, Born ratio | `sec5_7_theory/theory_closure/theory_p15_p14.py` | `p15_p14_{T,eta,beta,spec,summary}.txt` |
| 14 | Exp.5 (Sec 5.7/App.) | error floor / two-bandwidth | `sec5_7_theory/theory_closure/p0_platform.py`, `both_train.py`, `p0_both_eval.py`, `kscan_train.py`, `trscan_train.py`, `p0_trscan.py` | `p0eval.out`, `p0treval.out`, `exp5_both_summary.md` |
| 15 | Table breadth (Sec 5.9) | periodic Riesz | `sec5_9_breadth/riesz_periodic_check.py` | `riesz_periodic_result.txt` |
| 16 | Table breadth (Sec 5.9) | diffusion/Fisher/Allen-Cahn | `sec5_9_breadth/reaction_diffusion_check.py` | `reaction_diffusion_result.txt` |
| 17 | Table breadth (Sec 5.9) | 2-D vector Burgers | `sec5_9_breadth/system_burgers_check.py` | `system_burgers_result.txt` |
| 18 | Table breadth (Sec 5.9) | coupled system J | `sec5_9_breadth/system_coupled_check.py` | `system_coupled_result.txt` |
| 19 | Table breadth (Sec 5.9) | 2-D fractional NS | `sec5_9_breadth/ns_vorticity_check.py` | `ns_vorticity_result.txt` |
| 20 | Table breadth (Sec 5.9) | 3-D fractional NS | `sec5_9_breadth/ns3d_vorticity_check.py` | `ns3d_result.txt` |
| 21 | Appendix MMS table | convergence orders | `appendixA_code_verification/mms_convergence.py` | `mms_result.txt` |
| 22 | Fig. modal decay | analytic schematic | `figures/make_modal_decay_fig.py` | `figures/modal_decay_vs_T.pdf/.png` |
| 23 | Fig. B1 snapshot | solution field | `sec5_2_B1_main/plot_structure_b1.py` | `fig_snapshot_structure.png` |
| 24 | Fig. B1 radial spectrum | spectrum | `sec5_2_B1_main/plot_spectrum_b1.py` | `fig_radial_spectrum.png` |
| 25 | Fig. 1 architecture | hand-drawn (not generated) | — | not included in code repo |
| 26 | Algorithm 1 (offline table) | code | `frfno_core.build_prop_table_1d` | — |
| 27 | Algorithm 2 (training) | code | `frfno_core.DualNet` + `b1_burgers.train` | — |
| 28 | Algorithm 3 (inference) | code | `frfno_core.ub_full_batch/prop_at` + `DualNet.forward` | — |
| 29 | Table model parameter counts | code | `models_surrogate.n_params`, `models_extra` | printed by each benchmark |
| 30 | 1-D vs 2-D table regression | sanity | `sec5_2_B1_main/b1_frfno_regression.py`, `sec5_3_B2_longwindow/b2_frfno_regression.py` | `b1/b2_frfno_regression_result.txt` |

## Notes
* **Multi-script tables.** B1/B2/B4 rows are concatenations: the FrFNO-family
  columns (FrFNO/PDNO/DeepONet/U-Net) come from one script and the FNO/PINO/CNO
  columns from `rerun_3models.py`, because the competitor runs use a separate
  official-training policy. The two halves share the identical data, order grid,
  reference solver and evaluation tiers.
* **Weight dependencies.** `speedup_benchmark.py` and the B1 plotting scripts
  read `b1_weights.pt`, produced by `b1_burgers.py`; `run_all.py` orders stages
  accordingly. `*.pt`/large `*.npz` are regenerated, not committed.
* **Stale-column caveat.** Inside `b2b_longT_result.txt`, FNO/PINO/CNO columns
  predate the official rerun; for B2 competitors use the B2 block of
  `rerun_3models_result.txt`.
* **Random seeds.** test orders use `RandomState(2024)`; training/initial-field
  seeds are fixed literals inside each script, so runs are deterministic.
