"""rpgeom: exact limits of random projections for preserving geometry.

Companion library to "Exact Limits of Random Projections for Preserving
Geometry: Distance Recovery, Nearest-Neighbor Rankings, and Covariance
Shape in Gaussian Models" (P. Sao, 2026).

Quick start::

    from rpgeom import budget, audit
    print(budget(d=768, m=64, q=10))
"""

from . import ceilings, channels, ensembles, laws, maps, mmse
from .audit import AuditReport, audit
from .budget import BudgetReport, budget
from .recommend import DimensionRecommendation, recommend_dimension

__version__ = "0.1.0"

__all__ = [
    "audit",
    "AuditReport",
    "budget",
    "BudgetReport",
    "ceilings",
    "channels",
    "ensembles",
    "laws",
    "maps",
    "mmse",
    "recommend_dimension",
    "DimensionRecommendation",
]
