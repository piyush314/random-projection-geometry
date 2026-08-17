"""Theorem 8.3: Var(E[D|O]) <= 8 sum_{j<=m} lambda_j^2 for every rank-m map,
with equality at the top-m eigenspace.  Adversarial random sweep using the
exact Gaussian conditional-variance formula.  The filename retains the
number from the earlier manuscript layout."""

import numpy as np

from harness import FULL, check


def _var_cond(S, R):
    Cuu = 2 * R @ S @ R.T
    B = 2 * S @ R.T @ np.linalg.inv(Cuu)
    M = B.T @ B
    X = M @ Cuu
    return 2 * np.trace(X @ X)


@check("Thm 8.3", "spectral ceiling for the distance value, adversarial sweep")
def run():
    rng = np.random.default_rng(0)
    d, m = 8, 3
    trials = 4000 if FULL else 800
    worst = -np.inf
    for _ in range(trials):
        G = rng.standard_normal((d, d))
        S = G @ G.T / d
        lam = np.sort(np.linalg.eigvalsh(S))[::-1]
        bound = 8 * np.sum(lam[:m] ** 2)
        Q, _ = np.linalg.qr(rng.standard_normal((d, m)))
        worst = max(worst, _var_cond(S, Q.T) / bound)
    # attainment
    G = rng.standard_normal((d, d))
    S = G @ G.T / d
    lam, V = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    attained = _var_cond(S, V[:, :m].T)
    bound = 8 * np.sum(lam[:m] ** 2)
    return [
        (
            f"bound holds over {trials} random (Sigma, Pi)",
            worst <= 1.0 + 1e-9,
            f"max ratio {worst:.6f} <= 1",
        ),
        (
            "equality at top-m eigenspace",
            abs(attained / bound - 1) < 1e-9,
            f"{attained:.6f} vs {bound:.6f}",
        ),
    ]
