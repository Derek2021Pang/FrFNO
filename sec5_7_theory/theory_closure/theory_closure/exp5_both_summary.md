# Exp5 Supplementary: K and Training Grid Enlarged Together

## Purpose
Test Corollary cor:plateau's prediction that the error floor drops when
network modes K AND training-label bandwidth N0 are enlarged *together*.
The original Exp5 only had two axes: (a) vary K at fixed 17^2 labels,
(b) vary training grid at fixed K=10. This supplement adds the third axis:
K and N0 enlarged simultaneously.

## Code
- Training: theory_closure_20260909/both_train.py
  - both_train.py 16 32  -> both_K16_Nx32.pt  (FrFNO 281s + FNO 207s)
  - both_train.py 24 64  -> both_K24_Nx64.pt  (FrFNO 821s + FNO 658s)
- Evaluation: theory_closure_20260909/p0_both_eval.py
  - Frozen 513^2 truth (p0_ref513.npz), 48 samples, eval tiers 17/65/129/257
  - Results: p0_both_eval.npz (56 keys)

## Full Results: FrFNO rel.L2 error (%) by eval grid

| configuration        | K  | N0(train) | N=17  | N=65  | N=129 | N=257 (floor) |
|----------------------|----|-----------|-------|-------|-------|---------------|
| baseline             | 10 | 17^2      | 33.12 | 12.21 | 10.50 | 10.58         |
| K only               | 16 | 17^2      | 33.18 | 14.16 | 13.98 | 14.80         |
| K only               | 24 | 17^2      | 33.24 | 16.13 | 17.10 | 18.52         |
| grid only            | 10 | 33^2      | 32.93 | 10.18 | 7.82  | 7.75          |
| grid only            | 10 | 65^2      | 33.13 | 9.37  | 6.54  | 6.47          |
| **BOTH**             | 16 | 33^2      | 32.96 | 10.51 | 8.38  | 8.47          |
| **BOTH**             | 24 | 65^2      | 33.74 | 9.24  | 6.28  | 6.77          |

## Full Results: FNO rel.L2 error (%) by eval grid

| configuration        | K  | N0(train) | N=17  | N=65  | N=129 | N=257 (floor) |
|----------------------|----|-----------|-------|-------|-------|---------------|
| baseline             | 10 | 17^2      | 33.10 | 19.98 | 20.34 | 21.06         |
| K only               | 16 | 17^2      | 33.15 | 40.85 | 45.42 | 48.33         |
| K only               | 24 | 17^2      | 33.16 | 38.25 | 47.28 | 51.65         |
| grid only            | 10 | 33^2      | 33.15 | 13.15 | 13.90 | 15.74         |
| grid only            | 10 | 65^2      | 34.08 | 9.49  | 8.06  | 9.31          |
| **BOTH**             | 16 | 33^2      | 36.53 | 22.61 | 27.61 | 30.19         |
| **BOTH**             | 24 | 65^2      | 47.97 | 9.52  | 15.17 | 18.77         |

## Key Comparison at N=257 (plateau region)

### FrFNO
- K=16 axis: K-only=14.80%, grid-only=7.75%, BOTH=8.47%
  -> BOTH is lower than baseline (10.58%) but NOT lower than grid-only (+0.72%)
- K=24 axis: K-only=18.52%, grid-only=6.47%, BOTH=6.77%
  -> BOTH is lower than baseline (10.58%) but NOT lower than grid-only (+0.30%)

### FNO
- K=16 axis: K-only=48.33%, grid-only=15.74%, BOTH=30.19%
- K=24 axis: K-only=51.65%, grid-only=9.31%, BOTH=18.77%

## Conclusion

1. Widening K alone on fixed 17^2 labels RAISES the floor (FrFNO 10.58->14.80->18.52%).
   Consistent with "cannot decrease the floor".
2. Refining the training grid at fixed K=10 LOWERS the floor monotonically
   (10.58->7.75->6.47%).
3. Enlarging K and N0 together DOES lower the floor vs baseline (10.58->8.47->6.77%),
   but does NOT beat grid-only refinement. In this experimental range the training-label
   bandwidth N0 is the sole active bottleneck; K=10 already has enough spectral capacity.
   Enlarging K beyond the bottleneck adds parameters without useful high-mode signal,
   slightly increasing estimation variance.

## Implication for Corollary cor:plateau

The original wording "The floor moves down ONLY WHEN ... enlarged TOGETHER" is too strong.
Correct statement: the floor is controlled by the bottleneck min(K, N0); enlarging the
bottleneck side lowers it, while enlarging the non-bottleneck side alone cannot. A genuine
"together required" regime would need K=10 to itself become the bottleneck (N0 >= 129^2),
which is beyond the current training budget.
