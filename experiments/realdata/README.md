# Real-feature audit

Supply an observations-by-features matrix as `.npy`, `.csv`, or whitespace
text. The experiment records the spectrum-aware diagnostics and the measured
survival of pairwise and nearest-neighbor decisions under one seeded Haar
projection. Without `--input`, it runs a synthetic smoke example.

```bash
python experiments/realdata/run.py --input embeddings.npy --m 64 --q 10 \
  --output artifacts/my-audit.json
```

This is a diagnostic, not a universal guarantee: differences from the
Gaussian prediction are themselves informative about clusters, anisotropy,
or low intrinsic dimension.
