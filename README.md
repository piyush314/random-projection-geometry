# Random Projection Geometry

[![tests](https://github.com/piyush314/random-projection-geometry/actions/workflows/ci.yml/badge.svg)](https://github.com/piyush314/random-projection-geometry/actions/workflows/ci.yml)
[![theorem checks](https://github.com/piyush314/random-projection-geometry/actions/workflows/verify-theorems.yml/badge.svg)](https://github.com/piyush314/random-projection-geometry/actions/workflows/verify-theorems.yml)
[![notebooks](https://github.com/piyush314/random-projection-geometry/actions/workflows/notebooks.yml/badge.svg)](https://github.com/piyush314/random-projection-geometry/actions/workflows/notebooks.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-13213c.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-1e8ea1.svg)](pyproject.toml)

**Exact laws, practical diagnostics, and reproducible experiments for the
geometry hidden below a Johnson–Lindenstrauss guarantee.**

> Companion to *Exact Limits of Random Projections for Preserving Geometry:
> Distance Recovery, Nearest-Neighbor Rankings, and Covariance Shape in
> Gaussian Models* · Piyush Sao · paper link coming at public launch

A random projection can preserve every sampled distance to fixed relative
accuracy while losing the much smaller fluctuations that determine rankings,
nearest neighbors, and covariance shape. This repository turns that distinction
into tools you can calculate, teach, audit, and reproduce.

## Start with a decision, not an epsilon

```python
from rpgeom import budget, recommend_dimension

print(budget(d=768, m=64, q=10))
print(recommend_dimension(d=768, kendall_tau=0.30))
```

```text
Sketch budget: d = 768 -> m = 64   (m/d = 0.08333)
----------------------------------------------------------------
Distance values & every nonlinear feature (HGR ceiling) : 0.2887
Mutual information I(D; sketch)                          : 0.0436 nats
One pairwise comparison preserved (P)                    : 0.5929
Expected Kendall tau of a ranking                        : 0.1858
Nearest-of-10 preserved (Gaussian-score limit)           : 0.1946
Mean / log-scale information retained                    : 0.0833
Covariance *shape* information retained                  : 0.007040
```

Or audit an observations-by-features matrix:

```python
import numpy as np
from rpgeom import audit

X = np.load("embeddings.npy")
report = audit(X, m=64, q=10, n_trials=5000, rng=np.random.default_rng(7))
print(report)
```

The audit reports spectrum-aware diagnostics and compares exact isotropic
Gaussian predictions with measured survival under one seeded Haar projection.
Disagreement is useful evidence about anisotropy, clusters, tails, or low
intrinsic dimension—not something the tool hides.

## What is exact here?

For the Gaussian models stated in the paper:

- **One distance:** the decoder-free HGR ceiling is `sqrt(m/d)`, with the
  complete Laguerre singular spectrum and exact mutual information.
- **One comparison:** the survival probability follows an exact finite-`m,d`
  Beta–arcsine law; expected Kendall tau is `2p - 1`.
- **Nearest of q:** fixed-candidate overlap converges to an explicit Gaussian
  common-argmin (plurality-stability) kernel.
- **Covariance shape:** diffuse traceless shape information retains exactly
  `(m-1)(m+2)/((d-1)(d+2))`, versus `m/d` for mean and scale.
- **Anisotropy:** unequal spectral-block retention can create nonlinear excess;
  balanced fractional retention restores the clean ceiling.
- **General maps:** rank, spectral spread, and small singular values govern
  optimal inference, unwhitened norms, and noisy inversion, respectively.

These are model-specific theorems, not universal empirical claims. The APIs and
site label assumptions at the point of use.

## Install and validate

```bash
git clone https://github.com/piyush314/random-projection-geometry.git
cd random-projection-geometry
python -m pip install -e ".[dev]"
pytest -q
rpgeom verify
```

Python 3.10+; CPU only. Before the public PyPI release, the Git checkout is the
canonical installation.

## Five entry points

| Goal | Command or path |
|---|---|
| Compute a budget | `rpgeom budget --d 768 --m 64 --q 10` |
| Choose a dimension | `rpgeom recommend --d 768 --kendall-tau .30` |
| Audit data | `rpgeom audit embeddings.npy --m 64 --q 10` |
| Learn interactively | [`notebooks/tutorials/`](notebooks/tutorials/) |
| Reproduce the article | `rpgeom reproduce-paper --full` |

The [Quarto learning site](https://piyush314.github.io/random-projection-geometry/)
adds a browser calculator, nine guided lessons, a concept map, and a theorem-to-code index.

## Reproducibility architecture

```text
src/rpgeom/                 tested laws, diagnostics, reports, CLI
schemas/                    machine-readable report contracts
verification/               one independent check per theorem
reproduction/paper/         manuscript table and data generators
scripts/                    deterministic paper-asset generation
experiments/                four contract-driven experiment families
notebooks/tutorials/        nine executable lessons
notebooks/reproduction/     provenance walkthroughs
docs/                       Quarto site and browser calculator
```

Run the theorem ledger:

```bash
python verification/run_all.py
```

Run the archival bundle:

```bash
python experiments/paper_tables/run.py --full --output artifacts/paper
```

It writes machine-readable data, LaTeX rows, five PDFs, and a SHA-256 artifact
manifest. Exact identities, deterministic-equivalent proxies, asymptotic laws,
and Monte Carlo estimates are never presented as the same kind of evidence.

## CLI and JSON

All three report commands accept `--json`. Their output contracts live in
[`schemas/`](schemas/), making them suitable for CI gates and downstream tools.

```bash
rpgeom budget --d 400 --m 50 --q 8 --json
rpgeom recommend --d 400 --shape-information .10 --json
rpgeom audit matrix.csv --m 50 --trials 5000 --seed 42 --json
```

## Contributing and citation

Bug reports, numerical counterexamples, documentation corrections, and new
data-audit case studies are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
Please cite the paper using [CITATION.cff](CITATION.cff). Versioned public
releases are prepared for Zenodo archival.

MIT licensed. See [release_checklist.md](release_checklist.md) for the public
release gates.
