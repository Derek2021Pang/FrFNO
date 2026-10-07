# Section 5.9 — Breadth beyond Burgers

One-knife FrFNO-vs-FNO (and extra baselines where noted) transfers of the same
"exact frozen propagator + learned residual" construction:

| Script | Equation | Output |
|---|---|---|
| `riesz_periodic_check.py` | periodic Riesz fractional Laplacian (FFT symbol) | `riesz_periodic_result.txt` |
| `reaction_diffusion_check.py` | linear fractional diffusion, Fisher-KPP, Allen-Cahn | `reaction_diffusion_result.txt` |
| `system_burgers_check.py` | 2-D two-component vector fractional Burgers | `system_burgers_result.txt` |
| `system_coupled_check.py` | two-component system with non-symmetric coupling J | `system_coupled_result.txt` |
| `ns_vorticity_check.py` | 2-D incompressible fractional NS (vorticity) | `ns_vorticity_result.txt` |
| `ns3d_vorticity_check.py` | small 3-D fractional NS proof-of-concept | `ns3d_result.txt` |

Each is self-contained and checkpoints incrementally.
