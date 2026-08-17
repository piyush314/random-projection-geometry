#!/usr/bin/env python3
"""Reproduce the paper's numerical checks and anisotropy sweep.

Usage:
    python reproduction/paper/verify_claims.py --seed 42 --full

The script writes machine-readable CSV files and the LaTeX rows used by
``src/verification.tex``.  Exact checks have zero reported standard error;
Monte Carlo checks report a one-standard-error estimate.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.special import roots_jacobi


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "paper" / "data"

# Table 3 follows the restructured manuscript: distance recovery first,
# followed by rankings, covariance shape, and general-map/ensemble extensions.
TABLE_ROW_ORDER = (
    "isotropic_corr",
    "isotropic_mmse",
    "centered_error",
    "counterexample",
    "beta_arcsine",
    "shape",
    "single_chain",
    "two_stage_chain",
    "wishart_trace",
    "r2_proxy",
    "quarter_circle",
    "ensemble_average",
)


@dataclass
class Check:
    key: str
    label_tex: str
    predicted: float
    observed: float
    std_error: float
    method: str
    sample_count: int
    display_digits: int = 5
    is_identity: bool = True

    @property
    def z_score(self) -> float:
        if not self.is_identity:
            return math.nan
        if self.std_error == 0:
            return 0.0 if math.isclose(self.predicted, self.observed, rel_tol=0, abs_tol=1e-10) else math.inf
        return (self.observed - self.predicted) / self.std_error


def beta_arcsine_probability(m: int, d: int, nodes: int = 256) -> float:
    """Evaluate 1/2 + E[asin(sqrt(B))]/pi for B ~ Beta(m/2,(d-m)/2)."""
    a = m / 2
    b = (d - m) / 2
    x, w = roots_jacobi(nodes, b - 1, a - 1)
    values = np.arcsin(np.sqrt((x + 1) / 2))
    return 0.5 + float(np.dot(w, values) / np.sum(w)) / math.pi


def correlation_with_se(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    r = float(np.corrcoef(x, y)[0, 1])
    se = (1 - r * r) / math.sqrt(len(x) - 3)
    return r, se


def mean_with_se(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values)))


def odd_double_factorial_moment(k: int) -> float:
    """Return E[(G^2)^k] for G ~ N(0,1)."""
    if k == 0:
        return 1.0
    return float(math.prod(range(1, 2 * k, 2)))


def _distance_moment(power: int, lam: float) -> float:
    return sum(
        math.comb(power, a)
        * lam**a
        * odd_double_factorial_moment(a)
        * odd_double_factorial_moment(power - a)
        for a in range(power + 1)
    )


def _observed_moment(power: int, lam: float) -> float:
    return lam**power * odd_double_factorial_moment(power)


def _cross_moment(distance_power: int, observed_power: int, lam: float) -> float:
    return sum(
        math.comb(distance_power, a)
        * lam ** (a + observed_power)
        * odd_double_factorial_moment(a + observed_power)
        * odd_double_factorial_moment(distance_power - a)
        for a in range(distance_power + 1)
    )


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = eigh(matrix)
    tolerance = np.finfo(float).eps * len(values) * max(values[-1], 1.0)
    if values[0] <= tolerance:
        raise np.linalg.LinAlgError("Polynomial covariance matrix is numerically singular")
    return vectors @ np.diag(values**-0.5) @ vectors.T


def polynomial_cca(lam: float, degree: int) -> float:
    """Largest polynomial canonical correlation for D=lam*G1^2+G2^2, U=lam*G1^2."""
    powers = range(1, degree + 1)
    cov_dd = np.array(
        [
            [
                _distance_moment(i + j, lam)
                - _distance_moment(i, lam) * _distance_moment(j, lam)
                for j in powers
            ]
            for i in powers
        ]
    )
    cov_uu = np.array(
        [
            [
                _observed_moment(i + j, lam)
                - _observed_moment(i, lam) * _observed_moment(j, lam)
                for j in powers
            ]
            for i in powers
        ]
    )
    cov_du = np.array(
        [
            [
                _cross_moment(i, j, lam)
                - _distance_moment(i, lam) * _observed_moment(j, lam)
                for j in powers
            ]
            for i in powers
        ]
    )
    whitened = _inverse_sqrt(cov_dd) @ cov_du @ _inverse_sqrt(cov_uu)
    return float(np.linalg.svd(whitened, compute_uv=False)[0])


def anisotropy_sweep(points: int = 181, max_ratio: float = 10.0) -> list[dict[str, float]]:
    ratios = np.unique(np.concatenate((np.geomspace(1.0, max_ratio, points), [2.0])))
    rows: list[dict[str, float]] = []
    for lam in ratios:
        row = {
            "lambda_ratio": float(lam),
            "sqrt_alpha_1": float(lam / math.sqrt(lam * lam + 1)),
        }
        for degree in range(1, 5):
            row[f"cca_degree_{degree}"] = polynomial_cca(float(lam), degree)
        rows.append(row)
    return rows


def _chunked_gaussian_stage_correlation(
    rng: np.random.Generator,
    d: int,
    widths: Iterable[int],
    samples: int,
) -> tuple[float, float]:
    truth = rng.chisquare(d, samples)
    output = truth.copy()
    for width in widths:
        output *= rng.chisquare(width, samples) / width
    return correlation_with_se(truth, output)


def _wishart_statistics(
    rng: np.random.Generator,
    m: int,
    d: int,
    draws: int,
    batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    trace_m2: list[np.ndarray] = []
    effective_ranks: list[np.ndarray] = []
    for start in range(0, draws, batch_size):
        count = min(batch_size, draws - start)
        g = rng.standard_normal((count, m, d)) / math.sqrt(m)
        gram = g @ np.swapaxes(g, 1, 2)
        trace = np.trace(gram, axis1=1, axis2=2)
        trace_square = np.einsum("bij,bji->b", gram, gram)
        trace_m2.append(trace_square)
        effective_ranks.append(trace * trace / trace_square)
    return np.concatenate(trace_m2), np.concatenate(effective_ranks)


def _shape_contraction(
    rng: np.random.Generator,
    m: int,
    d: int,
    frames: int,
) -> np.ndarray:
    raw = rng.standard_normal((d, d))
    h = (raw + raw.T) / 2
    h -= np.trace(h) / d * np.eye(d)
    denominator = np.sum(h * h)
    values = np.empty(frames)
    for index in range(frames):
        q, _ = np.linalg.qr(rng.standard_normal((d, m)))
        compressed = q.T @ h @ q
        compressed -= np.trace(compressed) / m * np.eye(m)
        values[index] = np.sum(compressed * compressed) / denominator
    return values


def run_checks(seed: int, full: bool) -> list[Check]:
    rng = np.random.default_rng(seed)
    n = 400_000 if full else 50_000
    ensemble_n = 1_000_000 if full else 100_000
    map_draws = 3_000 if full else 400

    checks: list[Check] = []

    # Beta--arcsine law via its conditional bivariate-normal representation.
    m, d = 5, 50
    predicted = beta_arcsine_probability(m, d)
    beta = rng.beta(m / 2, (d - m) / 2, n)
    rho = np.sqrt(beta)
    x = rng.standard_normal(n)
    y = rho * x + np.sqrt(1 - beta) * rng.standard_normal(n)
    agreement = (x * y > 0).astype(float)
    observed, se = mean_with_se(agreement)
    checks.append(Check("beta_arcsine", r"Beta--arcsine $p_{5,50}$", predicted, observed, se, "MC, conditional Gaussian", n))

    # Isotropic channel: correlation, optimal MMSE, and centered Euclidean error.
    m, d = 10, 200
    u = rng.chisquare(m, n)
    v = rng.chisquare(d - m, n)
    truth = u + v
    observed, se = correlation_with_se(truth, u)
    checks.append(Check("isotropic_corr", r"Isotropic $\Corr(D,\widetilde D)$", math.sqrt(m / d), observed, se, "MC, independent chi-square split", n))

    residual_square = (v - (d - m)) ** 2 / (2 * d)
    observed, se = mean_with_se(residual_square)
    checks.append(Check("isotropic_mmse", r"Normalized minimum mean-square error", 1 - m / d, observed, se, "MC, conditional mean", n))

    centered_error = ((d / m) * (u - m) - (truth - d)) ** 2 / (2 * d)
    observed, se = mean_with_se(centered_error)
    checks.append(Check("centered_error", r"Centered estimation MSE / $\operatorname{Var}(D)$", (d - m) / m, observed, se, "MC, rescaled projection", n, display_digits=3))

    # Exact Gaussian-chain identities.
    observed, se = _chunked_gaussian_stage_correlation(rng, 200, [200], n)
    checks.append(Check("single_chain", r"One Gaussian stage ($200\to200$)", math.sqrt(200 / 402), observed, se, "MC, chi-square stage", n))

    observed, se = _chunked_gaussian_stage_correlation(rng, 300, [150, 100], n)
    chain_predicted = math.sqrt((2 / 300) / ((1 + 2 / 300) * (1 + 2 / 150) * (1 + 2 / 100) - 1))
    checks.append(Check("two_stage_chain", r"Gaussian chain ($300\to150\to100$)", chain_predicted, observed, se, "MC, independent chi-square stages", n))

    observed, se = _chunked_gaussian_stage_correlation(rng, 400, [100], ensemble_n)
    checks.append(Check("ensemble_average", r"Ensemble average ($km{=}100,d{=}400$)", math.sqrt(100 / 502), observed, se, "MC, stacked Gaussian map", ensemble_n))

    # Wishart trace moment and the ratio-of-moments proxy for r_2.
    trace_m2, effective_ranks = _wishart_statistics(rng, 20, 60, map_draws)
    observed, se = mean_with_se(trace_m2)
    checks.append(Check("wishart_trace", r"Wishart $\E\tr(M^2)$", 60 * 81 / 20, observed, se, "MC, Gaussian maps", map_draws, display_digits=3))
    observed, se = mean_with_se(effective_ranks)
    checks.append(Check("r2_proxy", r"\shortstack[l]{Ratio-of-moments proxy vs.\\Monte Carlo mean of $r_2$}", 20 * 60 / 81, observed, se, "MC; proxy is not an identity", map_draws, display_digits=3, is_identity=False))

    # Quarter-circle MMSE: formula versus independent quadrature.
    epsilon = 0.2
    predicted = epsilon / 2 * (math.sqrt(epsilon * epsilon + 4) - epsilon)
    integrand: Callable[[float], float] = lambda s: (
        epsilon * epsilon / (s * s + epsilon * epsilon)
        * math.sqrt(max(4 - s * s, 0))
        / math.pi
    )
    observed = quad(integrand, 0, 2, epsabs=1e-12, epsrel=1e-12)[0]
    checks.append(Check("quarter_circle", r"Quarter-circle mean posterior-variance fraction ($\varepsilon{=}0.2$)", predicted, observed, 0.0, "deterministic quadrature", 0, display_digits=6))

    # Haar shape contraction.
    shape_values = _shape_contraction(rng, 8, 40, map_draws)
    observed, se = mean_with_se(shape_values)
    predicted = (8 - 1) * (8 + 2) / ((40 - 1) * (40 + 2))
    checks.append(Check("shape", r"\shortstack[l]{Diffuse traceless covariance-shape\\contraction ($8/40$)}", predicted, observed, se, "MC, Haar frames", map_draws, display_digits=6))

    # Exact polynomial CCA counterexample.
    observed = polynomial_cca(2.0, 2)
    predicted = math.sqrt((246 + 2 * math.sqrt(201)) / 311)
    checks.append(Check("counterexample", r"Quadratic canonical correlation at $\lambda{=}2$", predicted, observed, 0.0, "exact Gaussian moments", 0, display_digits=6))

    return checks


def write_csv(path: Path, checks: list[Check], seed: int) -> None:
    fields = ["key", "predicted", "observed", "std_error", "z_score", "is_identity", "method", "sample_count", "seed"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for check in checks:
            writer.writerow(
                {
                    "key": check.key,
                    "predicted": f"{check.predicted:.12g}",
                    "observed": f"{check.observed:.12g}",
                    "std_error": f"{check.std_error:.12g}",
                    "z_score": "" if math.isnan(check.z_score) else f"{check.z_score:.6g}",
                    "is_identity": check.is_identity,
                    "method": check.method,
                    "sample_count": check.sample_count,
                    "seed": seed,
                }
            )


def write_anisotropy_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fields = ["lambda_ratio", "sqrt_alpha_1", "cca_degree_1", "cca_degree_2", "cca_degree_3", "cca_degree_4"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_results(checks: list[Check], rows: list[dict[str, float]], full: bool) -> None:
    """Fail loudly if an exact identity or calibrated Monte Carlo check regresses."""
    z_limit = 3.0 if full else 4.0
    failures = [
        f"{check.key}: |z|={abs(check.z_score):.3f} exceeds {z_limit}"
        for check in checks
        if check.is_identity and check.std_error > 0 and abs(check.z_score) > z_limit
    ]
    failures.extend(
        f"{check.key}: exact calculation did not match the prediction"
        for check in checks
        if check.is_identity and check.std_error == 0 and not math.isfinite(check.z_score)
    )

    isotropic = min(rows, key=lambda row: abs(row["lambda_ratio"] - 1.0))
    for degree in range(1, 5):
        if not math.isclose(
            isotropic[f"cca_degree_{degree}"],
            1 / math.sqrt(2),
            rel_tol=0,
            abs_tol=1e-10,
        ):
            failures.append(f"degree-{degree} isotropic CCA did not equal 1/sqrt(2)")

    for row in rows:
        if not math.isclose(
            row["cca_degree_1"], row["sqrt_alpha_1"], rel_tol=1e-10, abs_tol=1e-10
        ):
            failures.append(f"degree-one budget mismatch at lambda={row['lambda_ratio']}")
        values = [row[f"cca_degree_{degree}"] for degree in range(1, 5)]
        if any(right + 1e-9 < left for left, right in zip(values, values[1:])):
            failures.append(f"polynomial CCA was not nested at lambda={row['lambda_ratio']}")

    lambda_two = min(rows, key=lambda row: abs(row["lambda_ratio"] - 2.0))
    counterexample = math.sqrt((246 + 2 * math.sqrt(201)) / 311)
    if not math.isclose(
        lambda_two["cca_degree_2"], counterexample, rel_tol=0, abs_tol=1e-10
    ):
        failures.append("lambda=2 quadratic CCA did not reproduce the counterexample")

    if failures:
        raise RuntimeError("verification failed:\n  " + "\n  ".join(failures))


def _format_value(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _uncertainty_decimal_places(std_error: float, significant_digits: int = 2) -> int:
    """Return decimal places needed to show an uncertainty with fixed precision."""
    exponent = math.floor(math.log10(abs(std_error)))
    return max(0, significant_digits - 1 - exponent)


def write_latex_rows(path: Path, checks: list[Check]) -> None:
    checks_by_key = {check.key: check for check in checks}
    if set(checks_by_key) != set(TABLE_ROW_ORDER) or len(checks_by_key) != len(checks):
        raise ValueError("Table-row order must contain every check key exactly once")

    with path.open("w") as handle:
        handle.write("% Generated by reproduction/paper/verify_claims.py; do not edit by hand.\n")
        for key in TABLE_ROW_ORDER:
            check = checks_by_key[key]
            if check.std_error:
                decimal_places = _uncertainty_decimal_places(check.std_error)
                predicted = _format_value(check.predicted, decimal_places)
                observed = _format_value(check.observed, decimal_places)
                observed += r"\,$\pm$\," + _format_value(check.std_error, decimal_places)
            else:
                predicted = _format_value(check.predicted, check.display_digits)
                observed = _format_value(check.observed, check.display_digits)
            method = check.method
            if check.sample_count:
                count_tex = f"{check.sample_count:,}".replace(",", r"{,}")
                method += rf", $N={count_tex}$"
            handle.write(
                f"{check.label_tex} & {predicted} & {observed} & {method} \\\\\n"
            )
        handle.write("\\bottomrule\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Use the paper's full Monte Carlo sample sizes")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = run_checks(args.seed, args.full)
    anisotropy_rows = anisotropy_sweep()
    validate_results(checks, anisotropy_rows, args.full)
    csv_path = args.output_dir / "verification.csv"
    anisotropy_path = args.output_dir / "anisotropy.csv"
    latex_path = args.output_dir / "verification_table.tex"

    write_csv(csv_path, checks, args.seed)
    write_anisotropy_csv(anisotropy_path, anisotropy_rows)
    write_latex_rows(latex_path, checks)

    for check in checks:
        uncertainty = f" +/- {check.std_error:.3g}" if check.std_error else " (exact)"
        print(
            f"{check.key:20s} predicted={check.predicted:.8g} "
            f"observed={check.observed:.8g}{uncertainty} "
            + ("proxy comparison" if math.isnan(check.z_score) else f"z={check.z_score:.2f}")
        )
    for path in (csv_path, anisotropy_path, latex_path):
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        print(f"wrote {display_path}")


if __name__ == "__main__":
    main()
