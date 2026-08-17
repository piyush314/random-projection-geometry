"""General maps, chains, and ensembles (Theorems 11.1, 12.1, and A.1)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "chain_corr",
    "averaging_corr",
    "stacking_ceiling",
    "r2_matrix",
    "kantorovich_bounds",
    "rho_sigma",
]


def chain_corr(widths: list[int], d: int) -> float:
    """Corr(||Z||^2, ||P_L Z||^2) for a Gaussian chain, exact at every size.

    Appendix Theorem A.1: Corr^2 = (2/d) / (prod_{l=0..L} (1 + 2/n_l) - 1) with
    n_0 = d.  One stage of width m gives sqrt(m/(m+d+2)).
    """
    prod = np.prod([1.0 + 2.0 / n for n in [d] + list(widths)])
    return float(np.sqrt((2.0 / d) / (prod - 1.0)))


def averaging_corr(km: int, d: int) -> float:
    """Correlation of the averaged per-sketch estimate over k rank-m sketches.

    Theorem 12.1(ii): sqrt(km / (km + d + 2)), jointly over maps and data.
    """
    return float(np.sqrt(km / (km + d + 2)))


def stacking_ceiling(km: int, d: int) -> float:
    """HGR ceiling of the stacked (whitened) ensemble: sqrt(min(km, d)/d)."""
    return float(np.sqrt(min(km, d) / d))


def r2_matrix(M: np.ndarray) -> float:
    """Participation ratio r_2(M) = (tr M)^2 / tr(M^2)."""
    t1 = np.trace(M)
    t2 = np.trace(M @ M)
    return float(t1**2 / t2)


def kantorovich_bounds(singvals: np.ndarray) -> tuple[float, float, float]:
    """(lower, r_2, upper) for r_2(T^T T) from Theorem 11.1(ii).

    r * 4 kappa^2/(kappa^2+1)^2  <=  r_2  <=  r,  kappa = s_max/s_min over
    the positive singular values.
    """
    s = np.asarray(singvals, dtype=float)
    s = s[s > 0]
    r = len(s)
    a = s**2
    r2 = float(a.sum() ** 2 / (a**2).sum())
    kappa = s.max() / s.min()
    lower = r * 4 * kappa**2 / (kappa**2 + 1) ** 2
    return float(lower), r2, float(r)


def rho_sigma(M: np.ndarray, Sigma: np.ndarray) -> float:
    """Alignment diagnostic rho_Sigma(T)^2 from SM1: how well M = T^T T
    aligns with Sigma^2 -- r_2 alone carries no singular-vector information."""
    num = np.trace(M @ Sigma @ Sigma) ** 2
    den = np.trace(Sigma @ Sigma) * np.trace(M @ Sigma @ M @ Sigma)
    return float(num / den)
