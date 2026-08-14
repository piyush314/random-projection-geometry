"""Theorem 5.3 + Proposition 5.4: the fixed-q nearest-neighbor limit is the
Gaussian plurality-stability kernel p_q(sqrt(m/d)); small-rho slope c_q =
q^2 (q-1) a_q^2.  Quadrature vs paper value, kernel MC, finite-dim MC, and
numerical-slope checks including c_2 = 1/pi exactly."""

import numpy as np

from harness import FULL, check
from rpgeom.laws import plurality_kernel, plurality_kernel_mc, slope_cq


@check("Thm 5.3/5.4", "NN plurality kernel + slope constants")
def run():
    rng = np.random.default_rng(4)
    out = []
    rho = np.sqrt(20 / 10_000)
    pq = plurality_kernel(rho, 8)
    out.append(
        (
            "quadrature p_8(sqrt(20/1e4)) matches paper integral",
            abs(pq - 0.138324998) < 2e-6,
            f"{pq:.9f} vs 0.138324998",
        )
    )
    mc, se = plurality_kernel_mc(rho, 8, n=1_500_000 if FULL else 600_000, rng=rng)
    out.append(
        ("kernel MC agrees", abs(mc - pq) < 4.5 * se, f"MC {mc:.6f} +- {se:.6f}")
    )
    # finite-dimensional NN agreement sits near (slightly below) the limit
    q, d, m = 8, 2000, 20
    M = 40_000 if FULL else 16_000
    B = 2000
    hits = 0
    for i in range(0, M, B):
        X = rng.standard_normal((B, q + 1, d)).astype(np.float32)
        diff = X[:, 1:, :] - X[:, :1, :]
        Df = (diff.astype(np.float64) ** 2).sum(-1)
        Dp = (diff[:, :, :m].astype(np.float64) ** 2).sum(-1)
        hits += (Df.argmin(1) == Dp.argmin(1)).sum()
    pfd = hits / M
    sefd = np.sqrt(pfd * (1 - pfd) / M)
    out.append(
        (
            "finite-dim MC within Berry-Esseen distance of the kernel",
            abs(pfd - pq) < 0.02,
            f"finite {pfd:.5f} +- {sefd:.5f} vs limit {pq:.5f} (finite-m deviation expected)",
        )
    )
    c2 = slope_cq(2)
    out.append(("c_2 = 1/pi", abs(c2 - 1 / np.pi) < 1e-9, f"{c2:.9f} vs {1/np.pi:.9f}"))
    for q_ in (3, 4):
        cq = slope_cq(q_)
        eps = 0.02
        slope = (plurality_kernel(eps, q_) - 1 / q_) / eps
        out.append(
            (
                f"numerical slope matches c_{q_}",
                abs(slope - cq) / cq < 0.05,
                f"formula {cq:.5f}, numerical {slope:.5f}",
            )
        )
    return out
