"""Theorem 4.3: I(D; O) = h(chi2_d) - h(chi2_{d-r}), with limits
-(1/2)log(1-alpha) (proportional) and r/(2d) (fixed r).  The closed form is
cross-checked by an independent Monte Carlo estimator that uses only exact
chi-square densities."""

import numpy as np
from scipy import stats

from harness import FULL, check
from rpgeom.ceilings import mutual_info


@check("Thm 4.3", "exact mutual information and both limits")
def run():
    rng = np.random.default_rng(1)
    out = []
    n = 2_000_000 if FULL else 400_000
    for (r, d) in [(3, 7), (20, 60)]:
        U = rng.chisquare(r, n)
        V = rng.chisquare(d - r, n)
        est = stats.chi2.logpdf(V, d - r) - stats.chi2.logpdf(U + V, d)
        mi_mc, se = est.mean(), est.std() / np.sqrt(n)
        mi = mutual_info(r, d)
        out.append(
            (
                f"I(D;O) closed form vs density-MC at (r,d)=({r},{d})",
                abs(mi_mc - mi) < 5 * se + 1e-4,
                f"formula {mi:.5f}, MC {mi_mc:.5f} +- {se:.5f}",
            )
        )
    alpha = 0.3
    d = 40000
    lim = -0.5 * np.log(1 - alpha)
    val = mutual_info(int(alpha * d), d)
    out.append(
        (
            "proportional limit -(1/2)log(1-alpha)",
            abs(val - lim) < 2e-4,
            f"I = {val:.6f} vs limit {lim:.6f} at d = {d}",
        )
    )
    r, d = 3, 1_000_000
    ratio = mutual_info(r, d) * 2 * d / r
    out.append(
        (
            "fixed-r limit r/(2d)",
            abs(ratio - 1) < 1e-3,
            f"I * 2d/r = {ratio:.6f} -> 1",
        )
    )
    return out
