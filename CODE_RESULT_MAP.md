# Code ↔ numerical-result map (FrFNO paper)

Every table/figure/algorithm in the manuscript, the script that generates it, and
the authoritative output file. Paths are relative to the package root. All FrFNO
numbers use the **one-dimensional** propagator table
(`frfno_core.build_prop_table_1d`).

Run status: **all** numbers in the paper follow the 2026-10-06/07 rerun
(main benchmarks B1/B2/B4, theory Exp.1/2/5, cost, breadth ×6, ablations C1/B2,
Exp.3/4, B3 ×2, and the MMS verification from 10-05). The rerun used the two
2026-10-06 code changes: (i) the unified train/eval 1-D propagator table and
(ii) the Gamma(2−α) constant-factor fix. Batch result files live in the package
root (`ROOT`, matching `OUT_TXT = os.path.join(ROOT, ...)` inside each driver).

| # | Paper location | Kind | Generating script | Output file / artifact | Last rerun |
|---|---|---|---|---|---|
| 1 | Table B1 (Sec 5.2) | all 7 models | `b1_burgers.py` (root) | `b1_burgers_result.txt` | 2026-10-06 |
| 2 | Table B2 (Sec 5.3) | all 7 models | `b2b_longT.py` (root) | `b2b_longT_result.txt` | 2026-10-06 |
| 3 | Table B3 (Sec 5.4) | FrFNO / FNO / PINO λ-sweep | `sec5_4_B3_spacetime/b3_pino.py` | `b3_result_lam0.txt`, `b3_result_lam0.5.txt` | 2026-10-07 |
| 4 | Table B4 (Sec 5.5) | integer-order, all methods | `b4_integer.py` (root) | `b4_result.txt` | 2026-10-06 |
| 5 | Table per-query cost (Sec 5.6) | direct vs FrFNO timing | `sec5_6_speedup/speedup_benchmark.py` | `speedup_benchmark_result.txt` | 2026-10-06 |
| 6 | Table break-even / offline (Sec 5.6) | cost analysis | `sec5_6_speedup/speedup_benchmark.py` | `speedup_benchmark_result.txt` | 2026-10-06 |
| 7 | Fig. total wall-clock (Sec 5.6) | figure | `sec5_6_speedup/plot_speedup.py` | paper `figs/speedup_benchmark.pdf/.png` (parent `FrFNO-JSC` dir if present, else `./figures/`) | 2026-10-07 |
| 8 | Table C1 (Sec 5.8 / SM S3) | B1 component ablation | `sec5_8_ablation/c_ablation.py` | `c_ablation_result.txt`, `c_ablation.npz`, `c_weights.pt`; seed-wise errors for the rigor Seeds row in `c_ablation_seeds_result.txt` (extracted from `c_ablation.npz`) | 2026-10-07 |
| 9 | Table C2 (Sec 5.8 / SM S3) | B2 component ablation | `sec5_8_ablation/b2_ablation.py` | `b2_ablation_result.txt`, `b2_ablation.npz` | 2026-10-06 |
| 10 | Exp.1–2 (Sec 5.7) | theory verification (Exp.1 uses the frozen-$513^2$ protocol, same basis as Tables S4/S5; Exp.2 the $s$-scan) | `sec5_7_theory/theory_verify.py` | `theory_verify_result.txt` | 2026-10-07 |
| 11 | Exp.3–4 (Sec 5.7/App.) | slopes T, η, β, Born ratio | `sec5_7_theory/theory_closure/theory_p15_p14.py` | `theory_closure/p15_p14_{T,eta,beta,spec,summary}.txt` | 2026-10-06 |
| 12 | Exp.5 (Sec 5.7/App.) | error floor / two-bandwidth / spectral tail / K-bottleneck (note: `p0_trscan.log` is the training-resolution-axis scan at fixed $K$; `p0_both_eval.log` is the $K$-and-grid-enlarged-together scan; the SM grid column cites `p0_both_eval.log`) | `sec5_7_theory/theory_closure/p0_platform.py`, `both_train.py`, `p0_both_eval.py`, `kscan_train.py`, `trscan_train.py`, `p0_trscan.py`, `exp5_supplement.py`, `exp5_validate.py` | `theory_closure/p0_platform.log`, `p0_both_eval.log`, `p0_trscan.log`, `exp5_supp.log`, `exp5_validate.log`, `kscan_rerun_20261006.log`, `trscan32/64_rerun_20261006.log`, `both1632/2464_rerun_20261006.log`, `kscan_N64_rerun_20261006.log`, `p0_*.npz`, `exp5_*.npz` | 2026-10-06 |
| 13 | Table breadth (Sec 5.9) | periodic Riesz | `sec5_9_breadth/riesz_periodic_check.py` | `riesz_periodic_result.txt` | 2026-10-06 |
| 14 | Table breadth (Sec 5.9) | diffusion/Fisher/Allen-Cahn | `sec5_9_breadth/reaction_diffusion_check.py` | `reaction_diffusion_result.txt` | 2026-10-06 |
| 15 | Table breadth (Sec 5.9) | 2-D vector Burgers | `sec5_9_breadth/system_burgers_check.py` | `system_burgers_result.txt` | 2026-10-06 |
| 16 | Table breadth (Sec 5.9) | coupled system J | `sec5_9_breadth/system_coupled_check.py` | `system_coupled_result.txt` | 2026-10-06 |
| 17 | Table breadth (Sec 5.9) | 2-D fractional NS | `sec5_9_breadth/ns_vorticity_check.py` | `ns_vorticity_result.txt` | 2026-10-06 |
| 18 | Table breadth (Sec 5.9) | 3-D fractional NS | `sec5_9_breadth/ns3d_vorticity_check.py` | `ns3d_result.txt` | 2026-10-06 |
| 19 | Appendix MMS table | convergence orders + weak singularity | `appendixA_code_verification/mms_convergence.py` | `mms_result.txt` | 10-05 (confirmed no rerun) |
| 20 | Fig. modal decay | analytic schematic | `figures/make_modal_decay_fig.py` | `figures/modal_decay_vs_T.pdf/.png` | — |
| 21 | Fig. B1 snapshot | solution field | `sec5_2_B1_main/plot_structure_b1.py` | paper `figs/fig_snapshot_structure.png` (parent dir if present, else `./figures/`) | 2026-10-07 |
| 22 | Fig. B1 band filling | per-mode phase error | `sec5_2_B1_main/plot_band_filling.py` | paper `figs/fig_band_filling.png` (parent dir if present, else `./figures/`) | 2026-10-07 |
| 23 | Fig. 1 architecture | hand-drawn (not generated) | — | `../FrFNO-JSC/figs/{FrFNO,FNO}_architecture.pdf` | — |
| 24 | Algorithm 1 (offline table) | code | `frfno_core.build_prop_table_1d` | — | — |
| 25 | Algorithm 2 (training) | code | `frfno_core.DualNet` + `b1_burgers.train` | — | — |
| 26 | Algorithm 3 (inference) | code | `frfno_core.ub_full_batch/prop_at` + `DualNet.forward` | — | — |
| 27 | Table model parameter counts | code | `models_surrogate.n_params`, `models_extra` | printed by each benchmark | — |
| 28 | rollout mechanism diagnostic (supporting; not part of the paper) | per-step error decomposition, input-perturbation sensitivity, linear-propagator time-shift (memory-restart) error | `rollout_diagnose.py` (root) | `rollout_diagnose_result.txt` | 2026-10-07 |
| 29 | semigroup test (supporting; not part of the paper) | pure linear-table check $E(2\tau)u_0$ vs $E(\tau)E(\tau)u_0$ over $\alpha$, incl.\ integer-order limit | `semigroup_test.py` (root) | `semigroup_test_result.txt` | 2026-10-07 |
| 30 | table-shift test (supporting; not part of the paper) | two big steps 0→0.015→0.03 on one 400-step table; first- vs second-half kernel; integer-order limit exact | `table_shift_test.py` (root) | `table_shift_test_result.txt` | 2026-10-07 |

## Notes

* **Single-driver tables.** B1/B2/B4 rows are produced by one driver
  (`b1_burgers.py`, `b2b_longT.py`, `b4_integer.py`), each training and
  evaluating all seven models (FrFNO/FNO/PINO/PDNO/CNO/DeepONet/U-Net) on the
  identical data, order grid, reference solver and evaluation tiers.
* **Weight dependencies.** `speedup_benchmark.py` and the B1 plotting scripts
  read `b1_weights.pt`, produced by `b1_burgers.py`; `run_all.py` orders stages
  accordingly. `*.pt`/large `*.npz` are regenerated, not committed.
* **Output location.** The cost, breadth, B3 and ablation drivers write their
  `*_result.txt` to the package root (`OUT_TXT = os.path.join(ROOT, ...)`).
  Old copies under `sec5_9_breadth/`, `sec5_4_B3_spacetime/` and
  `sec5_8_ablation/` were deleted on 2026-10-07 so that only the rerun results
  remain.
* **Random seeds.** test orders use `RandomState(2024)`; training/initial-field
  seeds are fixed literals inside each script, so runs are deterministic.
* **C1 rerun fix.** `c_ablation.py` L141 was fixed on 2026-10-06
  (`json.dumps(row, default=float)`) to handle `np.float32`; the 19-configuration
  C1/C2/C3/C4 run completed on 2026-10-07.
* **Rerun logs.** All batch reruns (cost, breadth ×6, ablations, Exp.3/4, B3 ×2)
  are logged under `rerun_logs_20261006/`; see `RUNLOG_20261006.md`.
