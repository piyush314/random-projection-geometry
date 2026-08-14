"""Exact ranking laws under random projection (Theorems 5.1-5.4, SM3.1).

All formulas are for isotropic Gaussian point clouds in R^d sketched by a
rank-m projection (Haar frame or orthonormalized Gaussian map).  The three
central objects:

* ``pairwise_agreement(m, d)`` -- probability that one pairwise distance
  comparison survives projection (Beta-arcsine law, Theorem 5.1).
* ``plurality_kernel(rho, q)`` -- probability that the nearest of q
  candidates is preserved, in the Gaussian-score limit with correlation
  rho = sqrt(m/d) (Theorem 5.3).
* ``coupling_delta(q, m, d)`` -- the explicit nonasymptotic bound of
  Theorem SM3.1 controlling distance from chance level.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import integrate, special, stats

__all__ = [
    "a_q",
    "slope_cq",
    "pairwise_agreement",
    "kendall_tau",
    "plurality_kernel",
    "plurality_kernel_mc",
    "coupling_delta",
]


def a_q(q: int) -> float:
    """a_q = int phi(x)^2 Phi(x)^{q-2} dx  (Proposition 5.4)."""
    if q < 2:
        raise ValueError("q must be >= 2")
    val, _ = integrate.quad(
        lambda x: stats.norm.pdf(x) ** 2 * stats.norm.cdf(x) ** (q - 2), -10, 10
    )
    return val


def slope_cq(q: int) -> float:
    """Small-rho slope of the plurality kernel: c_q = q^2 (q-1) a_q^2.

    Sanity: ``slope_cq(2) == 1/pi`` (Sheppard).
    """
    return q**2 * (q - 1) * a_q(q) ** 2


def _arcsin_beta_expectation(m: float, d: float) -> float:
    """E[arcsin sqrt(B)] for B ~ Beta(m/2, (d-m)/2), robust for m << d."""
    a, b = m / 2.0, (d - m) / 2.0
    mu = a / (a + b)
    sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    lo = max(0.0, mu - 40 * sd)
    hi = min(1.0, mu + 40 * sd)
    f = lambda x: np.arcsin(np.sqrt(x)) * stats.beta.pdf(x, a, b)
    val, _ = integrate.quad(f, lo, hi, limit=400)
    # add the (numerically negligible unless m ~ d) outer pieces
    if lo > 0.0:
        val += integrate.quad(f, 0.0, lo, limit=200)[0]
    if hi < 1.0:
        val += integrate.quad(f, hi, 1.0, limit=200)[0]
    return val


def pairwise_agreement(m: int, d: int) -> float:
    """P(one distance comparison survives projection), Theorem 5.1.

    p_{m,d} = 1/2 + (1/pi) E[arcsin sqrt(B)],  B ~ Beta(m/2, (d-m)/2).

    Example: ``pairwise_agreement(5, 50) = 0.59829`` (paper Table 3).
    """
    if not (0 < m < d):
        raise ValueError("need 0 < m < d")
    return 0.5 + _arcsin_beta_expectation(m, d) / np.pi


def kendall_tau(m: int, d: int) -> float:
    """Expected Kendall tau of one query's ranking after projection.

    E[tau] = 2 p_{m,d} - 1 <= sqrt(m/d); for m << d this is
    approximately (2/pi) sqrt(m/d).
    """
    return 2.0 * pairwise_agreement(m, d) - 1.0


def plurality_kernel(rho: float, q: int, nodes: int = 60) -> float:
    """p_q(rho): P(common argmin of q correlated Gaussian score vectors).

    Computes q * E[ S(X,Y)^{q-1} ] over a standard bivariate normal pair
    with correlation rho, where S is the joint survival function
    (equation (5.1) of the paper). Tensor Gauss--Hermite quadrature gives
    roughly machine precision for q=2 and the paper's tabulated cases.

    Limits: p_q(0) = 1/q (chance), p_q(1) = 1.
    Under projection, rho = sqrt(m/d).
    """
    if q < 2:
        raise ValueError("q must be >= 2")
    if rho <= 0.0:
        return 1.0 / q
    if rho >= 1.0:
        return 1.0
    if not isinstance(nodes, (int, np.integer)) or nodes < 12:
        raise ValueError("nodes must be an integer >= 12")
    gh_x, gh_w = hermgauss(nodes)
    standard = np.sqrt(2.0) * gh_x
    x = np.repeat(standard, nodes)
    independent = np.tile(standard, nodes)
    y = rho * x + np.sqrt(1.0 - rho**2) * independent
    # Central symmetry gives S_rho(x,y) = Phi_rho(-x,-y).
    survival = stats.multivariate_normal.cdf(
        np.column_stack((-x, -y)),
        mean=np.zeros(2),
        cov=np.array([[1.0, rho], [rho, 1.0]]),
        maxpts=100_000,
        abseps=1e-12,
        releps=1e-12,
    )
    weights = np.repeat(gh_w, nodes) * np.tile(gh_w, nodes) / np.pi
    return float(q * np.dot(weights, survival ** (q - 1)))


def plurality_kernel_mc(
    rho: float, q: int, n: int = 1_000_000, rng: np.random.Generator | None = None
) -> tuple[float, float]:
    """Monte Carlo estimate of p_q(rho); returns (estimate, standard error)."""
    rng = np.random.default_rng() if rng is None else rng
    G = rng.standard_normal((n, q))
    H = rng.standard_normal((n, q))
    Gr = rho * G + np.sqrt(1.0 - rho**2) * H
    hit = (G.argmin(1) == Gr.argmin(1)).mean()
    return float(hit), float(np.sqrt(hit * (1 - hit) / n))


def coupling_delta(q: int, m: int, d: int) -> float:
    """Explicit nonasymptotic bound Delta_{q,m,r} of Theorem SM3.1.

    Bounds |P(N_1 = tilde-N_1) - 1/q| and the TV distance of the joint
    neighborhood pair from independent uniform subsets.  Deliberately
    conservative; -> 0 whenever m = o(d) at fixed q.
    """
    r = d - m
    if r < 2:
        raise ValueError("need d - m >= 2")
    a = np.log(2 * q)
    log_cr = -0.5 * np.log(12 * np.pi) + special.gammaln((r - 1) / 2) - special.gammaln(r / 2)
    cr = np.exp(log_cr)
    raw = (q * (q - 1) / 2) * cr * (12 * np.sqrt(m * a) + 16 * a)
    return float(min(1.0, raw))
