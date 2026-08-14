"""Theorem 7.1: rank / spectrum / precision trichotomy for general maps.
Kantorovich bounds on r_2(M) over random spectra; quarter-circle posterior
loss q(eps) closed form vs quadrature; small-eps hard-edge fraction."""

import numpy as np

from harness import FULL, check
from rpgeom.ensembles import kantorovich_bounds
from rpgeom.mmse import fraction_below_snr, quarter_circle_loss, quarter_circle_loss_quad


@check("Thm 7.1", "general-map trichotomy: Kantorovich + hard edge")
def run():
    rng = np.random.default_rng(6)
    trials = 5000 if FULL else 1500
    ok = True
    worst = ""
    for _ in range(trials):
        r = rng.integers(2, 30)
        s = np.exp(rng.normal(0, 1.0, r))
        lo, r2, hi = kantorovich_bounds(s)
        if not (lo - 1e-9 <= r2 <= hi + 1e-9):
            ok = False
            worst = f"violated: lo={lo:.4f}, r2={r2:.4f}, hi={hi:.4f}"
            break
    out = [
        (
            f"Kantorovich two-sided bound on r_2 over {trials} random spectra",
            ok,
            worst or "never violated",
        )
    ]
    e = 0.2
    qc, qq = quarter_circle_loss(e), quarter_circle_loss_quad(e)
    out.append(
        (
            "q(eps) closed form == quarter-circle quadrature (paper: 0.180998)",
            abs(qc - qq) < 1e-9 and abs(qc - 0.180998) < 1e-6,
            f"closed {qc:.6f}, quadrature {qq:.6f}",
        )
    )
    frac = fraction_below_snr(0.05)
    out.append(
        (
            "hard-edge fraction ~ (2/pi) eps at small eps",
            abs(frac - 2 * 0.05 / np.pi) / frac < 0.01,
            f"{frac:.6f} vs {2*0.05/np.pi:.6f}",
        )
    )
    return out
