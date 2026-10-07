# Section 5.3 — B2 long integration window (T=0.06)

| Script | Produces | Paper |
|---|---|---|
| `b2b_longT.py` (at package **root**, a shared data module + driver) | FrFNO/PDNO/DeepONet/U-Net at T=0.06, `b2b_longT_result.txt`, `b2b_weights.pt` | Table B2 FrFNO-family columns |
| `sec5_2_B1_main/rerun_3models.py` (B2 block) | FNO/PINO/CNO at T=0.06 | Table B2 competitor columns |
| `b2_frfno_regression.py` (here) | 1-D-table FrFNO regression for B2; `b2_frfno_regression_result.txt` | consistency check |

Note: inside `b2b_longT_result.txt` the FNO/PINO/CNO columns are stale values
left by an earlier run; the authoritative competitor numbers for B2 are the B2
block of `rerun_3models_result.txt`.
