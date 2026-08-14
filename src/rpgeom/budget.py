"""One-call answer to "what does sketching to m dimensions buy me?"

>>> from rpgeom import budget
>>> print(budget(d=768, m=64, q=10))
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import ceilings, laws

__all__ = ["budget", "BudgetReport"]


@dataclass
class BudgetReport:
    d: int
    m: int
    q: int | None = None
    hgr_ceiling: float = 0.0
    mutual_info_nats: float = 0.0
    pairwise_agreement: float = 0.0
    kendall_tau: float = 0.0
    nn_agreement: float | None = None
    nn_chance: float | None = None
    mean_ratio: float = 0.0
    shape_ratio: float = 0.0
    alpha_m: float | None = None
    spectrum_r2: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the report."""
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        L = [
            f"Sketch budget: d = {self.d} -> m = {self.m}   (m/d = {self.m/self.d:.4g})",
            "-" * 64,
            f"Distance values & every nonlinear feature (HGR ceiling) : {self.hgr_ceiling:.4f}",
            f"Mutual information I(D; sketch)                          : {self.mutual_info_nats:.4f} nats",
            f"One pairwise comparison preserved (P)                    : {self.pairwise_agreement:.4f}",
            f"Expected Kendall tau of a ranking                        : {self.kendall_tau:.4f}",
        ]
        if self.nn_agreement is not None:
            L.append(
                f"Nearest-of-{self.q} preserved (Gaussian-score limit)      "
                f" : {self.nn_agreement:.4f}   (chance {self.nn_chance:.4f})"
            )
        L += [
            f"Mean / log-scale information retained                    : {self.mean_ratio:.4f}",
            f"Covariance *shape* information retained                  : {self.shape_ratio:.6f}",
        ]
        if self.alpha_m is not None:
            L += [
                f"Spectrum-aware value ceiling alpha_m (best rank-m map)   : {self.alpha_m:.4f}",
                f"Effective rank r_2(Sigma) of your spectrum               : {self.spectrum_r2:.1f}",
            ]
        for n in self.notes:
            L.append(f"note: {n}")
        return "\n".join(L)


def budget(
    d: int,
    m: int,
    q: int | None = None,
    spectrum: np.ndarray | None = None,
) -> BudgetReport:
    """Compute the full exact-information budget of a rank-m sketch of R^d.

    Parameters
    ----------
    d, m : ambient and sketch dimension.
    q : optional candidate count for the nearest-neighbor line.
    spectrum : optional eigenvalues of the data covariance; adds the
        spectrum-aware distance-value ceiling alpha_m and r_2(Sigma).
    """
    if not isinstance(d, (int, np.integer)) or not isinstance(m, (int, np.integer)):
        raise TypeError("d and m must be integers")
    if not (2 <= d and 0 < m < d):
        raise ValueError("need integers satisfying 0 < m < d")
    if q is not None and (not isinstance(q, (int, np.integer)) or q < 2):
        raise ValueError("q must be an integer >= 2")
    if spectrum is not None:
        spectrum = np.asarray(spectrum, dtype=float)
        if spectrum.ndim != 1 or len(spectrum) != d:
            raise ValueError("spectrum must be a one-dimensional array of length d")
        if np.any(~np.isfinite(spectrum)) or np.any(spectrum < 0) or not np.any(spectrum > 0):
            raise ValueError("spectrum must contain finite, nonnegative values and at least one positive value")

    rep = BudgetReport(d=int(d), m=int(m), q=None if q is None else int(q))
    rep.hgr_ceiling = ceilings.hgr_isotropic(m, d)
    rep.mutual_info_nats = ceilings.mutual_info(m, d)
    rep.pairwise_agreement = laws.pairwise_agreement(m, d)
    rep.kendall_tau = laws.kendall_tau(m, d)
    if q is not None:
        rep.nn_agreement = laws.plurality_kernel(np.sqrt(m / d), q)
        rep.nn_chance = 1.0 / q
    rep.mean_ratio = ceilings.mean_ratio(m, d)
    rep.shape_ratio = ceilings.shape_ratio(m, d)
    if spectrum is not None:
        rep.alpha_m = ceilings.alpha_m(spectrum, m)
        rep.spectrum_r2 = ceilings.effective_rank(spectrum)
        theta = m / d
        rep.notes.append(
            f"balanced spectral thinning at theta = m/d would give exact ceiling sqrt(theta) = {np.sqrt(theta):.4f}"
        )
    rep.notes.append(
        "Gaussian-model laws; ceilings are decoder-free (no estimator, linear or not, can beat them)."
    )
    return rep
