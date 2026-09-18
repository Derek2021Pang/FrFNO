# Section 5.6 — Surrogate speed-up and break-even

| Script | Produces |
|---|---|
| `speedup_benchmark.py` | direct-solver vs FrFNO online wall-clock at 17²/33²/65²/129², per-query speed-up, offline cost, break-even count, total cost at 1000/5000 queries → `speedup_benchmark_result.txt` |
| `plot_speedup.py` | `speedup_benchmark.pdf/.png` (reads the result text; no training) |

`speedup_benchmark.py` loads `b1_weights.pt` from the package root; run
`b1_burgers.py` (stage sec5_2) first. CUDA is synchronised around every timing.
