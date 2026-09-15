# Section 5.2 — Main benchmark B1 (2-D space–time fractional Burgers)

Run from the package root (or via `python run_all.py --only sec5_2`).

| Script | Produces | Paper |
|---|---|---|
| `rerun_field_dim2.py` | FrFNO, PDNO, DeepONet, U-Net columns; `rerun_field_dim2.log` | Table B1 |
| `rerun_3models.py` | FNO/PINO (`field_dim=2`) and CNO (official AdamW+L1) for B1 **and** B2/B4; `rerun_3models_result.txt` | Tables B1/B2/B4 competitor columns |
| `b1_frfno_regression.py` | 1-D vs 2-D propagator-table FrFNO regression; `b1_frfno_regression_result.txt` | consistency check behind the 1-D table numbers |
| `plot_structure_b1.py`, `plot_spectrum_b1.py` | `fig_snapshot_structure.png`, `fig_radial_spectrum.png` | qualitative B1 figures |

The qualitative figures need `b1_weights.pt`, written by running the root module
`b1_burgers.py` (the `run_all.py` sec5_2 stage does this). The shipped
`*.log` / `*_result.txt` are the reference outputs from the paper machine.
