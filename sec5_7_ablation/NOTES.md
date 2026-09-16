# Section 5.7 — Component ablations

| Script | Produces | Paper |
|---|---|---|
| `c_ablation.py` | C1 B1 ablation: full / no analytic base / no dynamic fractional feature / no neighbour bases, plus seed/data/width/grid/lr sensitivity; `c_ablation_result.txt` | Table C1 |
| `b2_ablation.py` | C2 B2 (T=0.06) ablation incl. no-conditioning variant; checkpoint `b2_ablation.npz`; `b2_ablation_result.txt` | Table C2 |

The "direct / learn-full-field" configuration is intentionally not reported in
the paper (it is structurally identical to an FNO for FrFNO) but remains
available in the code.
