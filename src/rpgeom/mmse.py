"""Additive-noise recovery and the hard edge (Theorem 7.1(iii), SM1.1)."""

from __future__ import annotations

import numpy as np
from scipy import integrate

__all__ = ["direction_recovery", "quarter_circle_loss", "quarter_circle_loss_quad", "fraction_below_snr"]


def direction_recovery(s: np.ndarray, eps: float) -> np.ndarray:
    """Recovered variance fraction s^2/(s^2 + eps^2) per singular direction."""
    s = np.asarray(s, dtype=float)
    return s**2 / (s**2 + eps**2)


def quarter_circle_loss(eps: float) -> float:
    """Limiting average posterior loss q(eps) = (eps/2)(sqrt(eps^2+4) - eps)
    for a normalized square Gaussian map under additive noise eps."""
    return float(eps / 2 * (np.sqrt(eps**2 + 4) - eps))


def quarter_circle_loss_quad(eps: float) -> float:
    """Same quantity by quadrature against the quarter-circle law (cross-check)."""
    f = lambda s: (eps**2 / (s**2 + eps**2)) * np.sqrt(4 - s**2) / np.pi
    val, _ = integrate.quad(f, 0, 2)
    return float(val)


def fraction_below_snr(eps: float) -> float:
    """Fraction of directions with signal-to-noise ratio below one,
    approximately (2/pi) eps for small eps."""
    f = lambda s: np.sqrt(4 - s**2) / np.pi
    val, _ = integrate.quad(f, 0, min(eps, 2.0))
    return float(val)
