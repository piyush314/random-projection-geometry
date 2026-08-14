"""Theorem SM3.1 (nonasymptotic chance-neighborhood coupling): the density
constant c_r is exact (Fourier inversion attains the sup at the mode), the
E W_U bound is conservative, and the end-to-end coupling bound holds in a
direct finite-dimensional simulation."""

import numpy as np
from scipy import integrate, special

from harness import FULL, check
from rpgeom.laws import coupling_delta


@check("Thm SM3.1", "explicit coupling: c_r exactness + end-to-end simulation")
def run():
    rng = np.random.default_rng(9)
    out = []
    r = 10
    cr = (1 / np.sqrt(12 * np.pi)) * special.gamma((r - 1) / 2) / special.gamma(r / 2)
    f0 = (1 / (2 * np.pi)) * integrate.quad(
        lambda t: (1 + 12 * t**2) ** (-r / 2), -np.inf, np.inf
    )[0]
    out.append(
        (
            "density constant c_r/2 equals f(0) exactly (inversion, r=10)",
            abs(cr / 2 - f0) < 1e-12,
            f"{cr/2:.10f} = {f0:.10f}",
        )
    )
    for (q, m) in [(3, 2), (8, 20)]:
        a = np.log(2 * q)
        bound = 12 * np.sqrt(m * a) + 16 * a
        U = 2 * rng.chisquare(m, size=(100_000, q))
        W = (U.max(1) - U.min(1)).mean()
        out.append(
            (
                f"E W_U bound at (q,m)=({q},{m})",
                W <= bound,
                f"MC {W:.2f} <= bound {bound:.2f}",
            )
        )
    q, m, d = 3, 2, 5000 if FULL else 4000
    delta = coupling_delta(q, m, d)
    trials = 30_000 if FULL else 9_000
    B = 1000
    mis = 0
    agr = 0
    for i in range(0, trials, B):
        Z = rng.standard_normal((B, q + 1, d))
        Dif = Z[:, 1:, :] - Z[:, 0:1, :]
        U = (Dif[:, :, :m] ** 2).sum(2)
        V = (Dif[:, :, m:] ** 2).sum(2)
        N1 = np.argmin(U + V, 1)
        mis += (N1 != np.argmin(V, 1)).sum()
        agr += (N1 == np.argmin(U, 1)).sum()
    pm, pa = mis / trials, agr / trials
    out.append(
        (
            "coupling failure probability within Delta",
            pm <= delta,
            f"P(N1 != argmin V) = {pm:.4f} <= Delta = {delta:.4f}",
        )
    )
    out.append(
        (
            "|P(N1 = tilde-N1) - 1/q| within Delta",
            abs(pa - 1 / q) <= delta,
            f"|{pa:.4f} - {1/q:.4f}| <= {delta:.4f}",
        )
    )
    return out
