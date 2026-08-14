"""Dimension recommendations from explicit fine-geometry targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import ceilings, laws

__all__ = ["DimensionRecommendation", "recommend_dimension"]


@dataclass(frozen=True)
class DimensionRecommendation:
    """Smallest sketch dimension satisfying all requested Gaussian-model targets."""

    d: int
    m: int
    q: int | None
    retained_fraction: float
    targets: dict[str, float]
    achieved: dict[str, float]
    assumptions: str = "isotropic Gaussian data; nearest-neighbor target uses the fixed-q score limit"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        lines = [
            f"Recommended dimension: m = {self.m} of d = {self.d} ({self.retained_fraction:.2%})",
            "Targets and achieved values:",
        ]
        for name, target in self.targets.items():
            lines.append(f"  {name:22s} >= {target:.4f}   achieved {self.achieved[name]:.4f}")
        lines.append(f"Assumptions: {self.assumptions}")
        return "\n".join(lines)


def recommend_dimension(
    d: int,
    *,
    q: int | None = None,
    hgr_capacity: float | None = None,
    pairwise_agreement: float | None = None,
    kendall_tau: float | None = None,
    nearest_neighbor: float | None = None,
    shape_information: float | None = None,
) -> DimensionRecommendation:
    """Find the smallest ``m`` satisfying every requested recovery target.

    Targets are fractions or probabilities in ``[0, 1]``. At least one
    target is required. ``q`` is required when ``nearest_neighbor`` is set.
    The routine searches the exact finite-dimensional laws for ``m=1,...,d-1``.
    """
    if not isinstance(d, (int, np.integer)) or d < 2:
        raise ValueError("d must be an integer >= 2")
    targets = {
        name: value
        for name, value in {
            "hgr_capacity": hgr_capacity,
            "pairwise_agreement": pairwise_agreement,
            "kendall_tau": kendall_tau,
            "nearest_neighbor": nearest_neighbor,
            "shape_information": shape_information,
        }.items()
        if value is not None
    }
    if not targets:
        raise ValueError("provide at least one recovery target")
    if nearest_neighbor is not None and q is None:
        raise ValueError("q is required for a nearest_neighbor target")
    if q is not None and (not isinstance(q, (int, np.integer)) or q < 2):
        raise ValueError("q must be an integer >= 2")
    for name, value in targets.items():
        if not np.isfinite(value) or not (0 <= value <= 1):
            raise ValueError(f"{name} must lie in [0, 1]")

    def evaluate(m: int) -> dict[str, float]:
        values = {
            "hgr_capacity": ceilings.hgr_isotropic(m, d),
            "shape_information": ceilings.shape_ratio(m, d),
        }
        if "pairwise_agreement" in targets or "kendall_tau" in targets:
            values["pairwise_agreement"] = laws.pairwise_agreement(m, d)
            values["kendall_tau"] = 2 * values["pairwise_agreement"] - 1
        if "nearest_neighbor" in targets:
            values["nearest_neighbor"] = laws.plurality_kernel(np.sqrt(m / d), int(q))
        return values

    def passes(m: int) -> bool:
        values = evaluate(m)
        return all(values[name] >= target for name, target in targets.items())

    if not passes(d - 1):
        raise ValueError("no dimension m < d satisfies all requested targets")
    lo, hi = 1, d - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if passes(mid):
            hi = mid
        else:
            lo = mid + 1
    achieved = evaluate(lo)
    return DimensionRecommendation(
        d=d,
        m=lo,
        q=q,
        retained_fraction=lo / d,
        targets=targets,
        achieved={name: achieved[name] for name in targets},
    )
