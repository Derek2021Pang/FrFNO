# Section 5.5 — B4 integer-order limit

The driver is the shared module `b4_integer.py` at the package **root**
(`python b4_integer.py`, or stage sec5_4 in run_all.py): it evaluates every
method at (alpha,s)=(1,1) and on orders near-uniform in [0.9,1]. It writes
`b4_result.txt` (reference copy in this folder) and `b4_weights.pt`.
The FNO/PINO/CNO columns are also emitted by `sec5_2_B1_main/rerun_3models.py`.
