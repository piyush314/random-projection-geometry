"""Theorems 4.1-4.2: the HGR ceiling sqrt(r/d) of the beta-gamma channel and
the Laguerre singular spectrum.  Two exact-moment checks (no sampling):
polynomial CCA up to degree 4 equals sqrt(r/d), and the spectral variance
decomposition reproduces Var(E[T^2|U]) exactly."""

import numpy as np

from harness import check
from rpgeom.ceilings import laguerre_singvals
from rpgeom.channels import chi2_moment, poly_cca_maxcorr_UT


@check("Thm 4.1/4.2", "HGR ceiling and Laguerre spectrum, exact moments")
def run():
    r, d = 3, 7
    out = []
    cca = poly_cca_maxcorr_UT(r, d, degree=4)
    out.append(
        (
            "degree-4 polynomial CCA equals sqrt(r/d) (no nonlinear excess, isotropic)",
            abs(cca - np.sqrt(r / d)) < 1e-10,
            f"{cca:.12f} vs {np.sqrt(r/d):.12f}",
        )
    )
    # direct Var(E[T^2 | U]) via chi2 moments
    EU = [chi2_moment(r, p) for p in range(5)]
    b = 2 * (d - r)
    var_direct = (
        (EU[4] - EU[2] ** 2)
        + b**2 * (EU[2] - EU[1] ** 2)
        + 2 * b * (EU[3] - EU[2] * EU[1])
    )
    # spectral prediction: l1^2 |<T^2, psi1>|^2 + l2^2 |<T^2, psi2>|^2
    ET = [chi2_moment(d, p) for p in range(5)]
    c1sq = (ET[3] - ET[2] * ET[1]) ** 2 / (ET[2] - ET[1] ** 2)
    c2sq = (ET[4] - ET[2] ** 2) - c1sq
    l = laguerre_singvals(r, d, 2) ** 2
    var_spec = l[0] * c1sq + l[1] * c2sq
    out.append(
        (
            "Laguerre spectral decomposition reproduces Var(E[T^2|U])",
            abs(var_direct - var_spec) < 1e-8,
            f"direct {var_direct:.6f} = spectral {var_spec:.6f}",
        )
    )
    # ell_1 = sqrt(r/d) exactly
    out.append(
        (
            "ell_1 = sqrt(r/d)",
            abs(laguerre_singvals(r, d, 1)[0] - np.sqrt(r / d)) < 1e-12,
            f"{laguerre_singvals(r, d, 1)[0]:.12f}",
        )
    )
    return out
