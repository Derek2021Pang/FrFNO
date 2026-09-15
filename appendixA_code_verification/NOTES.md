# Appendix — Method of Manufactured Solutions

`mms_convergence.py` verifies the reference solver component by component:
* temporal test — discrete spatial eigenvalue, continuous Caputo L1 in time →
  expected global order 2-alpha;
* spatial test — discrete L1 in time, continuous eigenvalue/gradient → expected
  second-order spatial convergence.

Output: `mms_result.txt` (reference copy included).
