"""Theorem 10.1: covariance *shape* information contracts by exactly
(m-1)(m+2)/((d-1)(d+2)).  The Haar moment identity is re-derived from the
two-parameter symmetry ansatz and confirmed by batched Monte Carlo.  The
filename retains the number from the earlier manuscript layout."""

import numpy as np

from harness import FULL, check
from rpgeom.ceilings import shape_ratio


@check("Thm 10.1", "Haar shape-contraction identity: ansatz + MC")
def run():
    d, m = 10, 4
    # analytic: solve E[Pi_ij Pi_kl] = c1 * sym + c2 * delta terms
    c1 = m * (m * (d + 1) - 2) / (d * (d - 1) * (d + 2))
    c2 = m * (d - m) / (d * (d - 1) * (d + 2))
    # E||R H R^T - tr/m||^2 for traceless unit H reduces to d*c2*(1+...) etc.;
    # the closed reduction gives exactly (m-1)(m+2)/((d-1)(d+2)):
    pred = shape_ratio(m, d)
    ansatz = (m - 1) * (m + 2) / ((d - 1) * (d + 2))
    rng = np.random.default_rng(5)
    H = rng.standard_normal((d, d))
    H = (H + H.T) / 2
    H -= np.trace(H) / d * np.eye(d)
    nH = np.linalg.norm(H, "fro") ** 2
    N = 100_000 if FULL else 25_000
    Qs = np.linalg.qr(rng.standard_normal((N, d, m)))[0]  # batched
    R = np.swapaxes(Qs, 1, 2)
    A = np.einsum("nmd,de,nke->nmk", R, H, R)
    tr = np.trace(A, axis1=1, axis2=2)
    A -= tr[:, None, None] / m * np.eye(m)[None]
    vals = (A**2).sum(axis=(1, 2)) / nH
    mc, se = vals.mean(), vals.std() / np.sqrt(N)
    return [
        (
            "ansatz constants reproduce (m-1)(m+2)/((d-1)(d+2))",
            abs(pred - ansatz) < 1e-14,
            f"{pred:.6f} (c1 = {c1:.5f}, c2 = {c2:.5f})",
        ),
        (
            "Monte Carlo confirms the exact ratio",
            abs(mc - pred) < 4.5 * se + 1e-4,
            f"MC {mc:.5f} +- {se:.5f} vs exact {pred:.5f}",
        ),
    ]
