"""Decoder-free information budgets (Theorems 7.1, 8.1-8.3, 8.5, 10.1).

These are ceilings over *all* square-integrable decoders (Hirschfeld-
Gebelein-Renyi maximal correlation and Fisher-information ratios), not
guarantees for any particular estimator.  No decoder, linear or not, can
exceed them.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = [
    "hgr_isotropic",
    "alpha_m",
    "alpha_pi",
    "laguerre_singvals",
    "chi2_entropy",
    "mutual_info",
    "mean_ratio",
    "scale_ratio",
    "shape_ratio",
    "effective_rank",
]


def hgr_isotropic(m: int, d: int) -> float:
    """Maximal correlation of any feature of D with the sketch: sqrt(m/d).

    Theorem 7.1 (via Dembo-Kagan-Shepp).  Applies to every f(D) in L^2,
    hence to every nonlinear decoder.
    """
    _validate_dimensions(m, d)
    return float(np.sqrt(m / d))


def alpha_m(spectrum: np.ndarray, m: int) -> float:
    """alpha_m(Sigma) = sum of top-m lambda_j^2 / sum of all lambda_j^2.

    Theorem 8.3: Corr(D, g(O))^2 <= alpha_m for every rank-m linear map
    and every decoder g of the distance *value*; tight for the top-m
    eigenspace projection.
    """
    lam = _validate_spectrum(spectrum)
    if not isinstance(m, (int, np.integer)) or not (0 < m <= len(lam)):
        raise ValueError("m must be an integer between 1 and len(spectrum)")
    lam = np.sort(lam)[::-1]
    return float(np.sum(lam[:m] ** 2) / np.sum(lam**2))


def alpha_pi(lam: np.ndarray, r: np.ndarray, s: np.ndarray) -> float:
    """alpha_Pi(Sigma) for a commuting block projection.

    Blocks g have eigenvalue lam[g] with multiplicity r[g], of which s[g]
    directions are retained.  Theorem 8.5: if every s[g]/r[g] equals a
    common theta, the full nonlinear ceiling is exactly sqrt(theta) and
    the linear witness attains it.
    """
    lam, r, s = (np.asarray(x, dtype=float) for x in (lam, r, s))
    return float(np.sum(lam**2 * s) / np.sum(lam**2 * r))


def laguerre_singvals(r: int, d: int, kmax: int = 10) -> np.ndarray:
    """Singular values ell_k of the beta-gamma conditional-expectation
    operator: ell_k = sqrt( (r/2)_k / (d/2)_k )  (Theorem 8.1, Griffiths).

    ell_1 = sqrt(r/d) is the HGR ceiling; higher modes decay geometrically
    with ratio approaching r/d.
    """
    _validate_dimensions(r, d)
    if not isinstance(kmax, (int, np.integer)) or kmax < 1:
        raise ValueError("kmax must be a positive integer")
    ks = np.arange(1, kmax + 1)
    logp = special.gammaln(r / 2 + ks) - special.gammaln(r / 2)
    logq = special.gammaln(d / 2 + ks) - special.gammaln(d / 2)
    return np.exp(0.5 * (logp - logq))


def chi2_entropy(k: float) -> float:
    """Differential entropy of chi^2_k (nats)."""
    return float(
        k / 2 + np.log(2) + special.gammaln(k / 2) + (1 - k / 2) * special.digamma(k / 2)
    )


def mutual_info(r: int, d: int) -> float:
    """Exact I(D; O) = h(chi^2_d) - h(chi^2_{d-r}) in nats (Theorem 8.2).

    Limits: -(1/2) log(1 - r/d) for proportional r, and r/(2d) for fixed r
    as d grows.
    """
    _validate_dimensions(r, d)
    return chi2_entropy(d) - chi2_entropy(d - r)


def mean_ratio(m: int, d: int) -> float:
    """Fraction of mean-direction Fisher information retained: m/d."""
    _validate_dimensions(m, d)
    return m / d


def scale_ratio(m: int, d: int) -> float:
    """Fraction of log-scale Fisher information retained: m/d."""
    _validate_dimensions(m, d)
    return m / d


def shape_ratio(m: int, d: int) -> float:
    """Fraction of traceless covariance-*shape* information retained.

    Theorem 10.1: exactly (m-1)(m+2) / ((d-1)(d+2)) ~ (m/d)^2 -- shape pays
    a quadratic price where mean and scale pay a linear one.
    """
    _validate_dimensions(m, d)
    return (m - 1) * (m + 2) / ((d - 1) * (d + 2))


def effective_rank(spectrum: np.ndarray) -> float:
    """r_2(Sigma) = (sum lam)^2 / sum lam^2 -- the fluctuation scale of D."""
    lam = _validate_spectrum(spectrum)
    return float(np.sum(lam) ** 2 / np.sum(lam**2))


def _validate_dimensions(m: int, d: int) -> None:
    if not isinstance(m, (int, np.integer)) or not isinstance(d, (int, np.integer)):
        raise TypeError("dimensions must be integers")
    if not (0 < m < d):
        raise ValueError("need 0 < m < d")


def _validate_spectrum(spectrum: np.ndarray) -> np.ndarray:
    lam = np.asarray(spectrum, dtype=float)
    if lam.ndim != 1 or lam.size == 0:
        raise ValueError("spectrum must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(lam)) or np.any(lam < 0) or not np.any(lam > 0):
        raise ValueError("spectrum must contain finite, nonnegative values and at least one positive value")
    return lam
