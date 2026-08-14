"""The beta-gamma channel and exact-moment polynomial CCA.

The isotropic sketch reduces (Theorem 4.1) to observing U ~ chi^2_r out of
T = U + V with independent V ~ chi^2_{d-r}.  ``poly_cca_maxcorr`` computes
the maximal correlation restricted to polynomial features of a given degree
using *exact* chi-square moments -- no sampling, no noise.  This is the
engine behind the Theorem 4.4 counterexample and the Theorem 4.5 identity.
"""

from __future__ import annotations

import numpy as np

__all__ = ["beta_gamma_sample", "chi2_moment", "poly_cca_maxcorr_UT"]


def beta_gamma_sample(r: int, d: int, n: int, rng: np.random.Generator | None = None):
    """Sample (T, U, V) with U ~ chi^2_r, V ~ chi^2_{d-r}, T = U + V."""
    rng = np.random.default_rng() if rng is None else rng
    U = rng.chisquare(r, n)
    V = rng.chisquare(d - r, n)
    return U + V, U, V


def chi2_moment(k: float, p: int) -> float:
    """E[X^p] for X ~ chi^2_k:  prod_{j<p} (k + 2j)."""
    out = 1.0
    for j in range(p):
        out *= k + 2 * j
    return out


def _joint_moment_UT(a: int, b: int, r: int, d: int) -> float:
    """E[U^a T^b] with T = U + V:  sum_i C(b,i) E[U^{a+i}] E[V^{b-i}]."""
    from math import comb

    return sum(
        comb(b, i) * chi2_moment(r, a + i) * chi2_moment(d - r, b - i) for i in range(b + 1)
    )


def poly_cca_maxcorr_UT(r: int, d: int, degree: int = 4) -> float:
    """Max canonical correlation between polynomial features of U and of T,
    computed from exact moments.

    For the beta-gamma channel the maximal correlation over *all* features
    is sqrt(r/d), attained already by linear features (Theorem 4.1 +
    Griffiths' Laguerre system): this function returns sqrt(r/d) to machine
    precision at every degree, which is itself a nontrivial check.
    """
    degs = list(range(1, degree + 1))
    # centered feature moments
    mU = {p: chi2_moment(r, p) for p in range(2 * degree + 1)}
    mT = {p: chi2_moment(d, p) for p in range(2 * degree + 1)}

    def cU(p):  # E[U^p] centered handled via covariance formulas below
        return mU[p]

    Sxx = np.array([[mU[i + j] - mU[i] * mU[j] for j in degs] for i in degs])
    Syy = np.array([[mT[i + j] - mT[i] * mT[j] for j in degs] for i in degs])
    Sxy = np.array(
        [[_joint_moment_UT(i, j, r, d) - mU[i] * mT[j] for j in degs] for i in degs]
    )
    M = np.linalg.solve(Sxx, Sxy) @ np.linalg.solve(Syy, Sxy.T)
    eig = np.linalg.eigvals(M).real
    return float(np.sqrt(max(eig)))
