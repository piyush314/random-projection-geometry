# Zero-information JL experiment

This experiment compares three labeled finite-set embeddings of an isotropic
Gaussian cloud:

- a normalized Haar projection, represented by a fixed coordinate subspace;
- an i.i.d. Gaussian map;
- an independent Gaussian replacement cloud matched only at the population
  distance baseline.

It produces two views. The manuscript figure isolates the central contrast:
Haar and replacement embeddings can meet the same all-pairs relative-distance
threshold, while only the Haar distances remain correlated with the original
distance fluctuations. The repository dashboard additionally reports Kendall
correlation, nearest-neighbor agreement, distortion/source correlation, and a
shared-pair diagnostic.

```bash
python experiments/zero_information_jl/run.py \
  --profile smoke --output artifacts/zero_information_jl

python experiments/zero_information_jl/run.py \
  --profile full --output artifacts/zero_information_jl
```

The fixed-coordinate implementation reproduces the joint law of the original
and Haar-projected **distance matrices** for isotropic data. It is not a claim
about the coordinate-level joint law of `(X, R @ X)`.

The shared-pair correlation is estimated across repeated independent triples.
Computing off-diagonal products after centering the pairs from one realized
cloud would return the artificial identity `-1/(q-1)` and is deliberately not
used here.

The trial CSV also records the empirically calibrated replacement obtained by
matching the realized mean squared distance. Its mean mismatch is zero up to
floating-point error. `sharpness.csv` compares simulated failures of the
disjoint-pair ratio with the exact two-sided $F_{m,d}$ tail probability used in
the manuscript's necessity argument.
