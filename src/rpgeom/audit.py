"""Theory vs *your* data: predict, then measure, on a user-supplied matrix.

>>> import numpy as np
>>> from rpgeom import audit
>>> X = np.random.default_rng(0).standard_normal((2000, 256))
>>> print(audit(X, m=32, q=10))

The predictions are the paper's exact Gaussian-model laws with rho =
sqrt(m/d); the measurements sketch your rows with a Haar frame and count
what actually survives.  Agreement tells you the Gaussian budget is the
budget you are living under; deviation tells you where your data's
structure (anisotropy, clusters, low intrinsic dimension) changes the
story -- and the reported r_2 and alpha_m say in which direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import ceilings, laws
from .maps import haar_frame

__all__ = ["audit", "AuditReport"]


@dataclass
class AuditReport:
    n: int
    d: int
    m: int
    q: int
    r2: float
    alpha_m: float
    pred_pairwise: float
    meas_pairwise: float
    meas_pairwise_se: float
    pred_nn: float
    meas_nn: float
    meas_nn_se: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the report."""
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return "\n".join(
            [
                f"Audit: n = {self.n} points, d = {self.d} -> m = {self.m}, q = {self.q} candidates",
                "-" * 64,
                f"Your spectrum: effective rank r_2 = {self.r2:.1f} of {self.d};  "
                f"alpha_{self.m} = {self.alpha_m:.4f}",
                f"Pairwise comparison preserved : predicted {self.pred_pairwise:.4f}   "
                f"measured {self.meas_pairwise:.4f} +- {self.meas_pairwise_se:.4f}",
                f"Nearest-of-{self.q} preserved     : predicted {self.pred_nn:.4f}   "
                f"measured {self.meas_nn:.4f} +- {self.meas_nn_se:.4f}   (chance {1/self.q:.4f})",
                "predictions = exact isotropic-Gaussian laws at rho = sqrt(m/d); "
                "deviations reflect your data's spectrum and cluster structure",
            ]
        )


def audit(
    X: np.ndarray,
    m: int,
    q: int = 10,
    n_trials: int = 2000,
    rng: np.random.Generator | None = None,
) -> AuditReport:
    """Sketch the rows of X to m dimensions and measure what survives."""
    rng = np.random.default_rng(0) if rng is None else rng
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array with observations in rows")
    if np.any(~np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    n, d = X.shape
    if not (0 < m < d):
        raise ValueError("need 0 < m < d")
    if n < q + 2:
        raise ValueError("need at least q + 2 points")
    if q < 2:
        raise ValueError("q must be >= 2")
    if n_trials < 20:
        raise ValueError("n_trials must be at least 20")

    Xc = X - X.mean(0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    spectrum = sv**2 / max(n - 1, 1)

    R = haar_frame(d, m, rng)
    Y = X @ R.T

    # pairwise-order survival on random triples (query, a, b)
    idx = rng.integers(0, n, size=(n_trials, 3))
    ok = idx[:, 0] != idx[:, 1]
    ok &= idx[:, 0] != idx[:, 2]
    ok &= idx[:, 1] != idx[:, 2]
    idx = idx[ok]
    D1 = ((X[idx[:, 0]] - X[idx[:, 1]]) ** 2).sum(1)
    D2 = ((X[idx[:, 0]] - X[idx[:, 2]]) ** 2).sum(1)
    P1 = ((Y[idx[:, 0]] - Y[idx[:, 1]]) ** 2).sum(1)
    P2 = ((Y[idx[:, 0]] - Y[idx[:, 2]]) ** 2).sum(1)
    agree = ((D1 - D2) * (P1 - P2) > 0).astype(float)
    meas_p, se_p = float(agree.mean()), float(agree.std() / np.sqrt(len(agree)))

    # nearest-of-q survival on random query/candidate draws
    hits = []
    for _ in range(min(n_trials, 2000)):
        pick = rng.choice(n, size=q + 1, replace=False)
        query, cand = pick[0], pick[1:]
        Dq = ((X[cand] - X[query]) ** 2).sum(1)
        Pq = ((Y[cand] - Y[query]) ** 2).sum(1)
        hits.append(float(Dq.argmin() == Pq.argmin()))
    hits = np.asarray(hits)
    meas_nn, se_nn = float(hits.mean()), float(hits.std() / np.sqrt(len(hits)))

    return AuditReport(
        n=n,
        d=d,
        m=m,
        q=q,
        r2=ceilings.effective_rank(spectrum),
        alpha_m=ceilings.alpha_m(spectrum, m),
        pred_pairwise=laws.pairwise_agreement(m, d),
        meas_pairwise=meas_p,
        meas_pairwise_se=se_p,
        pred_nn=laws.plurality_kernel(np.sqrt(m / d), q),
        meas_nn=meas_nn,
        meas_nn_se=se_nn,
    )
