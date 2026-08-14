"""Theorem 7.2: ensembles.  Averaging k rank-m sketches has correlation
exactly sqrt(km/(km+d+2)) (chi-square stage representation MC), the stacked
ceiling is sqrt(min(km,d)/d), and the worst averaging/stacking gap
approaches 1/sqrt(2) at km = d."""

import numpy as np

from harness import FULL, check
from rpgeom.ensembles import averaging_corr, stacking_ceiling


@check("Thm 7.2", "ensemble averaging constant + 1/sqrt(2) gap")
def run():
    rng = np.random.default_rng(7)
    km, d = 100, 400
    N = 2_000_000 if FULL else 600_000
    D = rng.chisquare(d, N)
    Q = D * rng.chisquare(km, N) / km
    c = np.corrcoef(D, Q)[0, 1]
    pred = averaging_corr(km, d)
    out = [
        (
            "MC correlation matches sqrt(km/(km+d+2))",
            abs(c - pred) < 0.004,
            f"MC {c:.5f} vs exact {pred:.5f}",
        )
    ]
    dd = 100_000
    gap = averaging_corr(dd, dd) / stacking_ceiling(dd, dd)
    out.append(
        (
            "averaging/stacking gap -> 1/sqrt(2) at km = d",
            abs(gap - 1 / np.sqrt(2)) < 1e-4,
            f"ratio {gap:.6f} vs {1/np.sqrt(2):.6f}",
        )
    )
    return out
