"""Theorem 5.1: pairwise agreement is the Beta-arcsine law.  Quadrature vs
paper Table 3/4 values, and quadrature vs an honest finite-dimensional Monte
Carlo with per-sample Haar frames."""

import numpy as np

from harness import FULL, check
from rpgeom.laws import kendall_tau, pairwise_agreement


@check("Thm 5.1", "Beta-arcsine pairwise law: table values + finite-dim MC")
def run():
    out = []
    p = pairwise_agreement(5, 50)
    out.append(("p_{5,50} matches Table 3", abs(p - 0.59829) < 5e-6, f"{p:.6f} vs 0.59829"))
    t = kendall_tau(2000, 1_000_000)
    out.append(
        ("E[tau] at (m,d)=(2000,1e6) matches Table 4", abs(t - 0.028476) < 5e-6, f"{t:.6f} vs 0.028476")
    )
    rng = np.random.default_rng(2)
    d, m = 50, 5
    N = 200_000 if FULL else 60_000
    agree = 0
    B = 20_000
    for i in range(0, N, B):
        n = min(B, N - i)
        X = rng.standard_normal((n, 3, d))
        Qs = np.linalg.qr(rng.standard_normal((n, d, m)))[0]
        D1 = ((X[:, 0] - X[:, 1]) ** 2).sum(-1)
        D2 = ((X[:, 0] - X[:, 2]) ** 2).sum(-1)
        P1 = (np.einsum("nd,ndm->nm", X[:, 0] - X[:, 1], Qs) ** 2).sum(-1)
        P2 = (np.einsum("nd,ndm->nm", X[:, 0] - X[:, 2], Qs) ** 2).sum(-1)
        agree += ((D1 - D2) * (P1 - P2) > 0).sum()
    phat = agree / N
    se = np.sqrt(phat * (1 - phat) / N)
    out.append(
        (
            "finite-dimensional MC matches the law",
            abs(phat - p) < 4.5 * se,
            f"MC {phat:.5f} +- {se:.5f} vs exact {p:.5f}",
        )
    )
    return out
