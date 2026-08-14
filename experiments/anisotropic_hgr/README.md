# Anisotropic nonlinear-recovery sweep

The isotropic HGR ceiling does not automatically become
`sqrt(alpha_m(Sigma))` under anisotropy. This deterministic experiment
computes polynomial canonical-correlation lower bounds for the paper's
two-eigenvalue family and marks where they exceed the distance-value bound.

```bash
python experiments/anisotropic_hgr/run.py --output artifacts/anisotropic_hgr.csv
```
