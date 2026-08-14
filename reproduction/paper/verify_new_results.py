#!/usr/bin/env python3
"""Verify the balanced-block, kNN, and JL/Kendall additions.

This script complements ``verify_claims.py`` without changing the manuscript's
generated LaTeX table.  It writes four machine-readable CSV files:

* ``balanced_block_verification.csv``: exact-moment polynomial CCA;
* ``pq_quadrature.csv``: deterministic Gauss--Hermite evaluation of p_q;
* ``knn_verification.csv``: finite-dimensional kNN Monte Carlo with SEs;
* ``jl_kendall_verification.csv``: simultaneous JL/Kendall Monte Carlo.

Use ``--full`` for the archival sample sizes.  The default is a faster smoke
test with the same checks and output schema.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.special import gammaln
from scipy.stats import kendalltau, multivariate_normal, norm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "paper" / "data"


def _chi_square_moment(df: int, power: int) -> float:
    """Return E[X**power] for X ~ chi-square(df)."""
    if power == 0:
        return 1.0
    return math.exp(
        power * math.log(2.0)
        + gammaln(df / 2 + power)
        - gammaln(df / 2)
    )


def _compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    for values in itertools.product(range(total + 1), repeat=parts):
        if sum(values) == total:
            yield values


def _mixed_block_moment(
    distance_power: int,
    u1_power: int,
    u2_power: int,
    ranks: tuple[int, int],
    retained: tuple[int, int],
    eigenvalues: tuple[float, float],
) -> float:
    """E[D**p U1**a U2**b] for independent chi-square block energies.

    Here D = lambda_1 (U1 + V1) + lambda_2 (U2 + V2).  A global factor
    in the squared distance is immaterial to canonical correlation.
    """
    s1, s2 = retained
    r1, r2 = ranks
    dfs = (s1, r1 - s1, s2, r2 - s2)
    weights = (eigenvalues[0], eigenvalues[0], eigenvalues[1], eigenvalues[1])
    extra_powers = (u1_power, 0, u2_power, 0)
    numerator = math.factorial(distance_power)
    total = 0.0
    for powers in _compositions(distance_power, 4):
        multinomial = numerator
        term = 1.0
        for index, power in enumerate(powers):
            multinomial /= math.factorial(power)
            term *= weights[index] ** power
            term *= _chi_square_moment(dfs[index], power + extra_powers[index])
        total += multinomial * term
    return total


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = eigh(matrix)
    tolerance = np.finfo(float).eps * len(values) * max(float(values[-1]), 1.0)
    if values[0] <= tolerance:
        raise np.linalg.LinAlgError("feature covariance is numerically singular")
    return vectors @ np.diag(values**-0.5) @ vectors.T


def balanced_block_exact_cca(
    theta: float,
    ranks: tuple[int, int] = (80, 120),
    eigenvalues: tuple[float, float] = (4.0, 1.0),
) -> dict[str, float | int]:
    """Exact-moment CCA of {D,D^2} against all degree-at-most-two U features."""
    retained = tuple(int(round(theta * rank)) for rank in ranks)
    realized = tuple(s / r for s, r in zip(retained, ranks))
    if not math.isclose(realized[0], realized[1], rel_tol=0, abs_tol=1e-14):
        raise ValueError("theta must give the same integer retention fraction in each block")

    distance_powers = (1, 2)
    observed_powers = ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    moment = lambda p, a, b: _mixed_block_moment(  # noqa: E731
        p, a, b, ranks, retained, eigenvalues
    )

    mean_x = np.array([moment(p, 0, 0) for p in distance_powers])
    mean_y = np.array([moment(0, a, b) for a, b in observed_powers])
    cov_xx = np.array(
        [
            [moment(p + q, 0, 0) - mean_x[i] * mean_x[j]
             for j, q in enumerate(distance_powers)]
            for i, p in enumerate(distance_powers)
        ]
    )
    cov_yy = np.array(
        [
            [moment(0, a + c, b + d) - mean_y[i] * mean_y[j]
             for j, (c, d) in enumerate(observed_powers)]
            for i, (a, b) in enumerate(observed_powers)
        ]
    )
    cov_xy = np.array(
        [
            [moment(p, a, b) - mean_x[i] * mean_y[j]
             for j, (a, b) in enumerate(observed_powers)]
            for i, p in enumerate(distance_powers)
        ]
    )

    # Standardization before whitening avoids scale-dependent conditioning.
    scale_x = np.sqrt(np.diag(cov_xx))
    scale_y = np.sqrt(np.diag(cov_yy))
    corr_xx = cov_xx / np.outer(scale_x, scale_x)
    corr_yy = cov_yy / np.outer(scale_y, scale_y)
    corr_xy = cov_xy / np.outer(scale_x, scale_y)
    whitened = _inverse_sqrt(corr_xx) @ corr_xy @ _inverse_sqrt(corr_yy)
    exact_cca = float(np.linalg.svd(whitened, compute_uv=False)[0])

    alpha = (
        eigenvalues[0] ** 2 * retained[0]
        + eigenvalues[1] ** 2 * retained[1]
    ) / (
        eigenvalues[0] ** 2 * ranks[0]
        + eigenvalues[1] ** 2 * ranks[1]
    )
    return {
        "rank_block_1": ranks[0],
        "rank_block_2": ranks[1],
        "retained_block_1": retained[0],
        "retained_block_2": retained[1],
        "lambda_block_1": eigenvalues[0],
        "lambda_block_2": eigenvalues[1],
        "theta": realized[0],
        "sqrt_alpha": math.sqrt(alpha),
        "theory_hgr": math.sqrt(realized[0]),
        "exact_moment_cca_degree_2": exact_cca,
        "absolute_error": abs(exact_cca - math.sqrt(realized[0])),
    }


def p_q_quadrature(q: int, rho: float, nodes: int = 60) -> float:
    """Evaluate p_q(rho) = q E[S_rho(X,Y)^(q-1)] deterministically.

    S_rho is the joint bivariate-normal survival function.  By central
    symmetry, S_rho(x,y) = Phi_rho(-x,-y), which lets us use the lower-tail
    CDF without mislabelling it as the CDF in the mathematical formula.
    """
    if q < 2 or not -1 < rho < 1:
        raise ValueError("require q >= 2 and -1 < rho < 1")
    gh_x, gh_w = hermgauss(nodes)
    standard = math.sqrt(2.0) * gh_x
    x = np.repeat(standard, nodes)
    independent = np.tile(standard, nodes)
    y = rho * x + math.sqrt(1.0 - rho * rho) * independent
    points = np.column_stack((-x, -y))
    survival = multivariate_normal.cdf(
        points,
        mean=np.zeros(2),
        cov=np.array([[1.0, rho], [rho, 1.0]]),
        maxpts=100_000,
        abseps=1e-12,
        releps=1e-12,
    )
    weights = np.repeat(gh_w, nodes) * np.tile(gh_w, nodes) / math.pi
    return float(q * np.dot(weights, survival ** (q - 1)))


def small_rho_coefficient(q: int) -> float:
    integrand = lambda x: norm.pdf(x) ** 2 * norm.cdf(x) ** (q - 2)
    integral = quad(integrand, -math.inf, math.inf, epsabs=1e-13, epsrel=1e-13)[0]
    return q * q * (q - 1) * integral * integral


def simulate_knn(
    rng: np.random.Generator,
    d: int,
    m: int,
    q: int,
    k: int,
    trials: int,
    batch_size: int = 10_000,
) -> tuple[float, float]:
    """Exact finite-dimensional simulation using conditional chi-square draws."""
    total = 0.0
    total_square = 0.0
    completed = 0
    while completed < trials:
        count = min(batch_size, trials - completed)
        query_observed = rng.chisquare(m, size=count)
        query_hidden = rng.chisquare(d - m, size=count)
        observed = rng.noncentral_chisquare(
            m, query_observed[:, None], size=(count, q)
        )
        hidden = rng.noncentral_chisquare(
            d - m, query_hidden[:, None], size=(count, q)
        )
        original = observed + hidden
        if k == 1:
            values = (
                np.argmin(observed, axis=1) == np.argmin(original, axis=1)
            ).astype(float)
        else:
            projected_set = np.argpartition(observed, k - 1, axis=1)[:, :k]
            original_set = np.argpartition(original, k - 1, axis=1)[:, :k]
            overlap = (
                projected_set[:, :, None] == original_set[:, None, :]
            ).any(axis=2).sum(axis=1)
            values = overlap / k
        total += float(values.sum())
        total_square += float(np.square(values).sum())
        completed += count
    estimate = total / trials
    sample_variance = max(
        0.0,
        (total_square - trials * estimate * estimate) / (trials - 1),
    )
    return estimate, math.sqrt(sample_variance / trials)


def _wishart_identity_bartlett(
    df: int, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    factor = np.zeros((dimension, dimension), dtype=float)
    for row in range(dimension):
        factor[row, row] = math.sqrt(rng.chisquare(df - row))
        if row:
            factor[row, :row] = rng.standard_normal(row)
    return factor @ factor.T


def _squared_distances(gram: np.ndarray) -> np.ndarray:
    diagonal = np.diag(gram)
    distances = diagonal[:, None] + diagonal[None, :] - 2.0 * gram
    np.maximum(distances, 0.0, out=distances)
    return distances


def _mean_querywise_kendall(original: np.ndarray, projected: np.ndarray) -> float:
    indices = np.arange(original.shape[0])
    values = []
    for query in indices:
        keep = indices != query
        values.append(
            kendalltau(original[query, keep], projected[query, keep]).statistic
        )
    return float(np.mean(values))


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def _wilson_interval(successes: int, trials: int, z: float = 1.95996398454) -> tuple[float, float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return center - half_width, center + half_width


def _beta_arcsine_kendall(m: int, d: int, tolerance: float = 1e-15) -> float:
    """Exact expected querywise Kendall correlation from the beta--arcsine law."""
    a = m / 2
    total = 0.0
    coefficient = 1.0
    for index in range(10_000):
        power = index + 0.5
        log_moment = (
            gammaln(a + power)
            - gammaln(a)
            + gammaln(d / 2)
            - gammaln(d / 2 + power)
        )
        term = coefficient * math.exp(log_moment)
        total += term
        if abs(term) <= tolerance * max(1.0, abs(total)):
            return (2.0 / math.pi) * total
        coefficient *= ((2 * index + 1) ** 2) / (
            2.0 * (index + 1) * (2 * index + 3)
        )
    raise RuntimeError("arcsine series failed to converge")


def simulate_jl_kendall(
    rng: np.random.Generator,
    n: int,
    d: int,
    m: int,
    trials: int,
    epsilon: float = 0.15,
    tau_threshold: float = 0.10,
) -> dict[str, float | int]:
    upper = np.triu_indices(n, 1)
    maximum_distortions = np.empty(trials)
    kendall_values = np.empty(trials)
    for trial in range(trials):
        observed_gram = _wishart_identity_bartlett(m, n, rng)
        hidden_gram = _wishart_identity_bartlett(d - m, n, rng)
        original = _squared_distances(observed_gram + hidden_gram)
        projected = _squared_distances((d / m) * observed_gram)
        relative_error = np.abs(projected[upper] / original[upper] - 1.0)
        maximum_distortions[trial] = float(relative_error.max())
        kendall_values[trial] = _mean_querywise_kendall(original, projected)

    jl_event = maximum_distortions <= epsilon
    low_tau_event = np.abs(kendall_values) <= tau_threshold
    coexistence = jl_event & low_tau_event
    max_mean, max_se = _mean_and_se(maximum_distortions)
    tau_mean, tau_se = _mean_and_se(kendall_values)

    result: dict[str, float | int] = {
        "n": n,
        "d": d,
        "m": m,
        "m_over_d": m / d,
        "trials": trials,
        "epsilon": epsilon,
        "tau_threshold": tau_threshold,
        "max_distortion_mean": max_mean,
        "max_distortion_se": max_se,
        "kendall_mean": tau_mean,
        "kendall_se": tau_se,
        "kendall_exact": _beta_arcsine_kendall(m, d),
    }
    for prefix, event in (("jl", jl_event), ("coexistence", coexistence)):
        successes = int(event.sum())
        rate = successes / trials
        lower, upper_bound = _wilson_interval(successes, trials)
        result[f"{prefix}_successes"] = successes
        result[f"{prefix}_rate"] = rate
        result[f"{prefix}_binomial_se"] = math.sqrt(rate * (1.0 - rate) / trials)
        result[f"{prefix}_wilson95_lower"] = lower
        result[f"{prefix}_wilson95_upper"] = upper_bound
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate(
    balanced_rows: list[dict[str, object]],
    quadrature_rows: list[dict[str, object]],
    jl_rows: list[dict[str, object]],
) -> None:
    failures = []
    for row in balanced_rows:
        if float(row["absolute_error"]) > 1e-9:
            failures.append(
                f"balanced-block CCA mismatch at theta={row['theta']}: "
                f"error={row['absolute_error']}"
            )

    q2 = next(row for row in quadrature_rows if row["case"] == "q2_arcsine")
    q2_exact = 0.5 + math.asin(float(q2["rho"])) / math.pi
    if abs(float(q2["p_q_quadrature"]) - q2_exact) > 2e-9:
        failures.append("q=2 quadrature did not recover the arcsine formula")
    for row in quadrature_rows:
        if float(row["rho"]) == 0.0:
            if abs(float(row["p_q_quadrature"]) - float(row["chance"])) > 2e-9:
                failures.append(f"p_q(0) did not equal 1/q for q={row['q']}")

    for row in jl_rows:
        z_score = (
            (float(row["kendall_mean"]) - float(row["kendall_exact"]))
            / float(row["kendall_se"])
        )
        if abs(z_score) > 4.5:
            failures.append(f"JL/Kendall mean missed exact expectation at m={row['m']}")
    if failures:
        raise RuntimeError("new-result verification failed:\n  " + "\n  ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--full", action="store_true", help="Use archival Monte Carlo sizes")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    balanced_rows = [
        balanced_block_exact_cca(theta) for theta in (0.10, 0.25, 0.50)
    ]

    quadrature_cases = [
        ("q2_arcsine", 2, 0.10),
        ("q8_independent", 8, 0.0),
        ("q8_d10000_m20", 8, math.sqrt(20 / 10_000)),
        ("q8_d100000_m20", 8, math.sqrt(20 / 100_000)),
        ("q16_d100000_m20", 16, math.sqrt(20 / 100_000)),
    ]
    quadrature_rows: list[dict[str, object]] = []
    quadrature_lookup: dict[tuple[int, float], float] = {}
    for case, q, rho in quadrature_cases:
        coefficient = small_rho_coefficient(q)
        value = p_q_quadrature(q, rho)
        quadrature_lookup[(q, rho)] = value
        quadrature_rows.append(
            {
                "case": case,
                "q": q,
                "rho": rho,
                "chance": 1 / q,
                "c_q": coefficient,
                "small_rho_prediction": 1 / q + coefficient * rho,
                "p_q_quadrature": value,
                "tail_function": "joint survival",
                "quadrature_method": "tensor Gauss-Hermite",
                "quadrature_nodes_per_axis": 60,
            }
        )

    knn_trials = 400_000 if args.full else 100_000
    knn_settings = [
        (10_000, 20, 8, 1),
        (100_000, 20, 8, 1),
        (100_000, 20, 16, 3),
    ]
    knn_rows: list[dict[str, object]] = []
    for index, (d, m, q, k) in enumerate(knn_settings):
        estimate, standard_error = simulate_knn(
            np.random.default_rng(args.seed + 100 + index),
            d,
            m,
            q,
            k,
            knn_trials,
        )
        rho = math.sqrt(m / d)
        gaussian_limit = quadrature_lookup.get((q, rho)) if k == 1 else None
        knn_rows.append(
            {
                "d": d,
                "m": m,
                "q": q,
                "k": k,
                "trials": knn_trials,
                "estimate": estimate,
                "standard_error": standard_error,
                "chance": k / q,
                "rho": rho,
                "gaussian_score_quadrature": "" if gaussian_limit is None else gaussian_limit,
                "finite_minus_gaussian_score": (
                    "" if gaussian_limit is None else estimate - gaussian_limit
                ),
                "seed": args.seed + 100 + index,
            }
        )

    jl_trials = 300 if args.full else 80
    jl_rows: list[dict[str, object]] = []
    for index, m in enumerate((1_000, 2_000, 5_000)):
        row = simulate_jl_kendall(
            np.random.default_rng(args.seed + 500 + index),
            n=100,
            d=1_000_000,
            m=m,
            trials=jl_trials,
        )
        row["kendall_z_score"] = (
            (float(row["kendall_mean"]) - float(row["kendall_exact"]))
            / float(row["kendall_se"])
        )
        row["seed"] = args.seed + 500 + index
        jl_rows.append(row)

    _validate(balanced_rows, quadrature_rows, jl_rows)
    outputs = (
        ("balanced_block_verification.csv", balanced_rows),
        ("pq_quadrature.csv", quadrature_rows),
        ("knn_verification.csv", knn_rows),
        ("jl_kendall_verification.csv", jl_rows),
    )
    for name, rows in outputs:
        path = args.output_dir / name
        _write_csv(path, rows)
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        print(f"wrote {display_path}")

    quarter = next(row for row in balanced_rows if row["theta"] == 0.25)
    print(
        "balanced theta=0.25: "
        f"theory={quarter['theory_hgr']:.12f}, "
        f"exact degree-2 CCA={quarter['exact_moment_cca_degree_2']:.12f}"
    )
    for row in knn_rows:
        print(
            f"kNN d={row['d']}, m={row['m']}, q={row['q']}, k={row['k']}: "
            f"{row['estimate']:.6f} +/- {row['standard_error']:.6f}"
        )
    for row in jl_rows:
        print(
            f"JL/Kendall m={row['m']}: JL={row['jl_rate']:.3f}, "
            f"coexistence={row['coexistence_rate']:.3f}, "
            f"tau={row['kendall_mean']:.5f} +/- {row['kendall_se']:.5f}"
        )


if __name__ == "__main__":
    main()
