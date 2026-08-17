"""Proposition 3.1: independent Gaussian replacement satisfies all pairwise
JL inequalities in the all-pairs concentration regime.  The check covers the
finite bound, exact empirical calibration, and the disjoint-pair F law used
for sharpness.  The filename retains the proposition number from the earlier
manuscript layout.
"""

import math

import numpy as np
from harness import FULL, check
from scipy.spatial.distance import pdist
from scipy.stats import f as f_distribution


@check("Prop 3.1", "independent replacement: JL bound, calibration, F ratio")
def run():
    rng = np.random.default_rng(15)
    n = 30 if FULL else 24
    d = 20_000 if FULL else 12_000
    m = 5000 if FULL else 3000
    epsilon = 0.5
    a = epsilon / (2 + epsilon)
    n_pairs = math.comb(n, 2)

    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, m)) * math.sqrt(d / m)
    original = pdist(x, "sqeuclidean")
    replacement = pdist(y, "sqeuclidean")
    max_error = float(np.max(np.abs(replacement / original - 1)))
    probability_lower_bound = 1 - 2 * n_pairs * math.exp(-d * a**2 / 8)
    probability_lower_bound -= 2 * n_pairs * math.exp(-m * a**2 / 8)

    calibrated = replacement * (original.mean() / replacement.mean())
    calibration_error = abs(calibrated.mean() / original.mean() - 1)
    rank_preserved = np.array_equal(np.argsort(calibrated), np.argsort(replacement))

    # Use moderate degrees of freedom here so the exact two-sided tail is
    # numerically visible; the all-pairs check above retains the paper scale.
    tail_m, tail_d, tail_epsilon = 64, 256, 0.2
    repetitions = 200_000 if FULL else 60_000
    ratio = (rng.chisquare(tail_m, repetitions) / tail_m) / (
        rng.chisquare(tail_d, repetitions) / tail_d
    )
    observed_tail = float(
        np.mean((ratio < 1 - tail_epsilon) | (ratio > 1 + tail_epsilon))
    )
    exact_tail = float(
        f_distribution.cdf(1 - tail_epsilon, tail_m, tail_d)
        + f_distribution.sf(1 + tail_epsilon, tail_m, tail_d)
    )

    return [
        (
            "the finite all-pairs event occurs where the stated bound is nonvacuous",
            probability_lower_bound > 0.99 and max_error <= epsilon,
            f"bound {probability_lower_bound:.6f}; max relative error {max_error:.4f}",
        ),
        (
            "empirical calibration matches the realized baseline and preserves ranks",
            calibration_error < 1e-14 and rank_preserved,
            f"relative mean mismatch {calibration_error:.2e}",
        ),
        (
            "disjoint-pair failures follow the exact F tail",
            abs(observed_tail - exact_tail) < 0.005,
            f"MC {observed_tail:.6f} vs exact {exact_tail:.6f}",
        ),
    ]
