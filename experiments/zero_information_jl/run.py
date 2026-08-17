#!/usr/bin/env python3
"""Compare a Haar sketch with an independent Gaussian replacement cloud."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402
from scipy.stats import f as f_distribution  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rpgeom.laws import kendall_tau  # noqa: E402
from rpgeom.plotting import COLORS, setup_style  # noqa: E402

PROFILES = {
    "smoke": {
        "d": 1024,
        "n": 50,
        "mgrid_a": np.array([32, 64, 128, 256, 512]),
        "trials_a": 8,
        "mgrid_c": np.array([32, 64, 128, 256, 512]),
        "trials_c": 5,
        "m_b": 256,
        "kernel_reps": 20_000,
        "shared_reps": 600,
        "sharpness_reps": 20_000,
    },
    "full": {
        "d": 8192,
        "n": 100,
        "mgrid_a": np.array([128, 181, 256, 362, 512, 724, 1024, 1448, 2048, 2896, 4096]),
        "trials_a": 30,
        "mgrid_c": np.array([128, 256, 512, 1024, 2048, 4096]),
        "trials_c": 15,
        "m_b": 2048,
        "kernel_reps": 200_000,
        "shared_reps": 3000,
        "sharpness_reps": 200_000,
    },
}

METHODS = ("haar", "gaussian", "replacement")
PAPER_METHODS = ("haar", "replacement")
LABELS = {"haar": "Haar", "gaussian": "i.i.d. Gaussian", "replacement": "replacement"}
COLORS_BY_METHOD = {
    "haar": COLORS["blue"],
    "gaussian": COLORS["medium_gray"],
    "replacement": COLORS["red"],
}


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def calibrate_replacement_distances(
    original: np.ndarray, replacement: np.ndarray
) -> np.ndarray:
    """Rescale replacement distances to match the realized pairwise mean."""
    return replacement * (original.mean() / replacement.mean())


def disjoint_pair_tail_diagnostic(
    rng: np.random.Generator,
    *,
    d: int,
    mgrid: np.ndarray,
    epsilon: float,
    repetitions: int,
) -> dict[str, np.ndarray]:
    """Compare simulated disjoint-pair ratio failures with exact F tails."""
    empirical = np.empty(len(mgrid))
    exact = np.empty(len(mgrid))
    for index, m_value in enumerate(mgrid):
        m = int(m_value)
        ratio = (rng.chisquare(m, repetitions) / m) / (
            rng.chisquare(d, repetitions) / d
        )
        empirical[index] = np.mean(
            (ratio < 1 - epsilon) | (ratio > 1 + epsilon)
        )
        exact[index] = f_distribution.cdf(1 - epsilon, m, d) + f_distribution.sf(
            1 + epsilon, m, d
        )
    return {"empirical": empirical, "exact": exact}


def _conditional_gaussian_coordinates(
    rng: np.random.Generator, points: np.ndarray, rows: int
) -> np.ndarray:
    """Sample unscaled Gaussian-map coordinates using only the n-by-n Gram matrix."""
    gram = points @ points.T
    jitter = np.finfo(float).eps * np.trace(gram) / len(gram)
    factor = np.linalg.cholesky(gram + jitter * np.eye(len(gram)))
    return rng.standard_normal((rows, len(points))) @ factor.T


def simulate_certificates(
    rng: np.random.Generator,
    *,
    d: int,
    n: int,
    mgrid: np.ndarray,
    trials: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Simulate the all-pairs certificate and two correlation diagnostics."""
    shape = (len(mgrid), trials)
    result = {
        metric: {method: np.empty(shape) for method in METHODS}
        for metric in ("max_error", "distance_corr", "distortion_corr")
    }
    result["calibrated_max_error"] = {"replacement": np.empty(shape)}
    result["calibrated_mean_error"] = {"replacement": np.empty(shape)}
    max_m = int(mgrid.max())
    for trial in range(trials):
        points = rng.standard_normal((n, d))
        truth = pdist(points, "sqeuclidean")
        replacement_base = rng.standard_normal((n, max_m))
        gaussian_base = _conditional_gaussian_coordinates(rng, points, max_m)
        for index, m_value in enumerate(mgrid):
            m = int(m_value)
            candidates = {
                "haar": (d / m) * pdist(points[:, :m], "sqeuclidean"),
                "gaussian": pdist(gaussian_base[:m].T, "sqeuclidean") / m,
                "replacement": (d / m)
                * pdist(replacement_base[:, :m], "sqeuclidean"),
            }
            for method, embedded in candidates.items():
                distortion = embedded / truth - 1.0
                result["max_error"][method][index, trial] = np.max(np.abs(distortion))
                result["distance_corr"][method][index, trial] = _safe_corr(embedded, truth)
                result["distortion_corr"][method][index, trial] = _safe_corr(distortion, truth)
            calibrated = calibrate_replacement_distances(
                truth, candidates["replacement"]
            )
            result["calibrated_max_error"]["replacement"][index, trial] = np.max(
                np.abs(calibrated / truth - 1.0)
            )
            result["calibrated_mean_error"]["replacement"][index, trial] = abs(
                calibrated.mean() / truth.mean() - 1.0
            )
    return result


def simulate_scatter_case(
    rng: np.random.Generator, *, d: int, n: int, m: int
) -> dict[str, np.ndarray | float]:
    points = rng.standard_normal((n, d))
    truth = pdist(points, "sqeuclidean")
    haar = (d / m) * pdist(points[:, :m], "sqeuclidean")
    replacement = (d / m) * pdist(rng.standard_normal((n, m)), "sqeuclidean")

    def standardize(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / values.std()

    z_truth = standardize(truth)
    z_haar = standardize(haar)
    z_replacement = standardize(replacement)
    return {
        "z_truth": z_truth,
        "z_haar": z_haar,
        "z_replacement": z_replacement,
        "rho_haar": _safe_corr(z_truth, z_haar),
        "rho_replacement": _safe_corr(z_truth, z_replacement),
        "max_error_haar": float(np.max(np.abs(haar / truth - 1.0))),
        "max_error_replacement": float(np.max(np.abs(replacement / truth - 1.0))),
    }


def simulate_rankings(
    rng: np.random.Generator,
    *,
    d: int,
    n: int,
    mgrid: np.ndarray,
    trials: int,
) -> dict[str, dict[str, np.ndarray]]:
    shape = (len(mgrid), trials)
    result = {
        metric: {method: np.empty(shape) for method in PAPER_METHODS}
        for metric in ("kendall", "nearest")
    }
    indices = np.arange(n)
    max_m = int(mgrid.max())
    for trial in range(trials):
        points = rng.standard_normal((n, d))
        original = squareform(pdist(points, "sqeuclidean"))
        replacement_base = rng.standard_normal((n, max_m))
        for index, m_value in enumerate(mgrid):
            m = int(m_value)
            matrices = {
                "haar": squareform((d / m) * pdist(points[:, :m], "sqeuclidean")),
                "replacement": squareform(
                    (d / m) * pdist(replacement_base[:, :m], "sqeuclidean")
                ),
            }
            for method, embedded in matrices.items():
                taus: list[float] = []
                hits = 0
                for query in indices:
                    candidates = indices[indices != query]
                    taus.append(
                        float(
                            kendalltau(
                                original[query, candidates], embedded[query, candidates]
                            ).statistic
                        )
                    )
                    hits += int(
                        candidates[np.argmin(embedded[query, candidates])]
                        == candidates[np.argmin(original[query, candidates])]
                    )
                result["kendall"][method][index, trial] = np.mean(taus)
                result["nearest"][method][index, trial] = hits / n
    return result


def plurality_kernel_mc_grid(
    rng: np.random.Generator,
    rho: np.ndarray,
    q: int,
    repetitions: int,
    batch_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    hits = np.zeros(len(rho), dtype=np.int64)
    completed = 0
    while completed < repetitions:
        count = min(batch_size, repetitions - completed)
        first = rng.standard_normal((count, q))
        second = rng.standard_normal((count, q))
        first_min = np.argmin(first, axis=1)
        for index, correlation in enumerate(rho):
            coupled = correlation * first + math.sqrt(1 - correlation**2) * second
            hits[index] += np.count_nonzero(first_min == np.argmin(coupled, axis=1))
        completed += count
    estimate = hits / repetitions
    standard_error = np.sqrt(estimate * (1 - estimate) / repetitions)
    return estimate, standard_error


def shared_pair_correlations(
    rng: np.random.Generator,
    *,
    d: int,
    mgrid: np.ndarray,
    repetitions: int,
    batch_size: int = 100,
) -> dict[str, np.ndarray]:
    """Estimate Corr(e_01,e_02) across repeated independent triples.

    Correlation is taken over repetitions.  Centering all pairs within one
    realized cloud would force the spurious value -1/(q-1).
    """
    max_m = int(mgrid.max())
    first = {method: np.empty((len(mgrid), repetitions)) for method in PAPER_METHODS}
    second = {method: np.empty((len(mgrid), repetitions)) for method in PAPER_METHODS}
    completed = 0
    scale = d / mgrid[None, :]
    while completed < repetitions:
        count = min(batch_size, repetitions - completed)
        points = rng.standard_normal((count, 3, d))
        diff_01 = points[:, 0] - points[:, 1]
        diff_02 = points[:, 0] - points[:, 2]
        truth_01 = np.sum(diff_01**2, axis=1)
        truth_02 = np.sum(diff_02**2, axis=1)
        observed_01 = np.cumsum(diff_01**2, axis=1)[:, mgrid - 1]
        observed_02 = np.cumsum(diff_02**2, axis=1)[:, mgrid - 1]

        replacement = rng.standard_normal((count, 3, max_m))
        rep_01 = np.cumsum((replacement[:, 0] - replacement[:, 1]) ** 2, axis=1)[
            :, mgrid - 1
        ]
        rep_02 = np.cumsum((replacement[:, 0] - replacement[:, 2]) ** 2, axis=1)[
            :, mgrid - 1
        ]
        selection = slice(completed, completed + count)
        first["haar"][:, selection] = (
            scale * observed_01 / truth_01[:, None] - 1.0
        ).T
        second["haar"][:, selection] = (
            scale * observed_02 / truth_02[:, None] - 1.0
        ).T
        first["replacement"][:, selection] = (
            scale * rep_01 / truth_01[:, None] - 1.0
        ).T
        second["replacement"][:, selection] = (
            scale * rep_02 / truth_02[:, None] - 1.0
        ).T
        completed += count
    return {
        method: np.array(
            [_safe_corr(first[method][index], second[method][index]) for index in range(len(mgrid))]
        )
        for method in PAPER_METHODS
    }


def _plot_band(ax, x: np.ndarray, values: np.ndarray, *, label: str, color: str) -> None:
    median = np.median(values, axis=1)
    lower, upper = np.quantile(values, [0.1, 0.9], axis=1)
    ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
    ax.plot(x, median, color=color, linewidth=2, marker="o", markersize=3, label=label)


def _format_ratio_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", visible=False)


def make_paper_figure(
    output: Path,
    *,
    d: int,
    mgrid: np.ndarray,
    epsilon: float,
    certificates: dict[str, dict[str, np.ndarray]],
    scatter: dict[str, np.ndarray | float],
    m_b: int,
) -> None:
    setup_style()
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )
    fig = plt.figure(figsize=(7.2, 2.75), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.35, 1, 1), wspace=0.08)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    ratio = mgrid / d
    for method in PAPER_METHODS:
        _plot_band(
            axes[0],
            ratio,
            certificates["max_error"][method],
            label=LABELS[method],
            color=COLORS_BY_METHOD[method],
        )
    axes[0].axhline(epsilon, color=COLORS["dark_gray"], linestyle="--", linewidth=1)
    axes[0].text(ratio[0], epsilon + 0.012, rf"JL threshold $\varepsilon={epsilon:.1f}$", fontsize=8)
    axes[0].set_title("(a) Uniform certificate")
    axes[0].set_xlabel(r"compression ratio  $m/d$")
    axes[0].set_ylabel("maximum relative error")
    axes[0].legend(loc="upper right")
    _format_ratio_axis(axes[0])

    z_truth = np.asarray(scatter["z_truth"])
    limit = 3.4
    for axis, method, title in (
        (axes[1], "haar", "(b) Haar"),
        (axes[2], "replacement", "(c) Replacement"),
    ):
        values = np.asarray(scatter[f"z_{method}"])
        color = COLORS_BY_METHOD[method]
        cmap = LinearSegmentedColormap.from_list(f"{method}_density", ["#ffffff", color])
        axis.hexbin(z_truth, values, gridsize=30, mincnt=1, cmap=cmap, linewidths=0)
        rho = float(scatter[f"rho_{method}"])
        line = np.linspace(-limit, limit, 100)
        axis.plot(line, rho * line, color=COLORS["dark_gray"], linewidth=1, linestyle="--")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("original distance (standardized)")
        axis.text(
            0.04,
            0.96,
            rf"$\rho={rho:.2f}$" + "\n" + rf"max err. $={float(scatter[f'max_error_{method}']):.2f}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
        )
        axis.grid(False)
    axes[1].set_ylabel("embedded distance (standardized)")
    axes[2].set_ylabel("")
    axes[2].text(
        0.96,
        0.04,
        rf"$m/d={m_b/d:.2f}$",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_dashboard(
    output: Path,
    *,
    d: int,
    n: int,
    epsilon: float,
    mgrid_a: np.ndarray,
    mgrid_c: np.ndarray,
    certificates: dict[str, dict[str, np.ndarray]],
    rankings: dict[str, dict[str, np.ndarray]],
    tau_theory: np.ndarray,
    nn_theory: np.ndarray,
    nn_theory_se: np.ndarray,
    shared: dict[str, np.ndarray],
) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.8), constrained_layout=True)
    axes = axes.ravel()
    ratio_a = mgrid_a / d
    ratio_c = mgrid_c / d
    for method in METHODS:
        _plot_band(
            axes[0],
            ratio_a,
            certificates["max_error"][method],
            label=LABELS[method],
            color=COLORS_BY_METHOD[method],
        )
    axes[0].axhline(epsilon, color=COLORS["dark_gray"], linestyle="--", linewidth=1)
    axes[0].set_title("A · Uniform JL certificate")
    axes[0].set_ylabel("maximum relative error")
    axes[0].legend()

    for method in PAPER_METHODS:
        mean = certificates["distance_corr"][method].mean(axis=1)
        axes[1].plot(ratio_a, mean, color=COLORS_BY_METHOD[method], marker="o", label=LABELS[method])
    axes[1].plot(ratio_a, np.sqrt(ratio_a), color=COLORS["blue"], linestyle="--", linewidth=1, label=r"$\sqrt{m/d}$")
    axes[1].axhline(0, color=COLORS["red"], linestyle="--", linewidth=1)
    axes[1].set_title("B · Distance signal")
    axes[1].set_ylabel(r"$\mathrm{Corr}(D,D_{\rm embedded})$")
    axes[1].legend()

    for method in PAPER_METHODS:
        values = rankings["kendall"][method]
        axes[2].errorbar(
            ratio_c,
            values.mean(axis=1),
            yerr=values.std(axis=1, ddof=1) / math.sqrt(values.shape[1]),
            color=COLORS_BY_METHOD[method],
            marker="o",
            capsize=2,
            label=LABELS[method],
        )
    axes[2].plot(ratio_c, tau_theory, color=COLORS["blue"], linestyle="--", linewidth=1, label="Haar theory")
    axes[2].axhline(0, color=COLORS["red"], linestyle="--", linewidth=1)
    axes[2].set_title("C · Mean Kendall correlation")
    axes[2].set_ylabel(r"mean $\tau$")
    axes[2].legend()

    for method in PAPER_METHODS:
        values = rankings["nearest"][method]
        axes[3].errorbar(
            ratio_c,
            values.mean(axis=1),
            yerr=values.std(axis=1, ddof=1) / math.sqrt(values.shape[1]),
            color=COLORS_BY_METHOD[method],
            marker="o",
            capsize=2,
            label=LABELS[method],
        )
    axes[3].errorbar(ratio_c, nn_theory, yerr=nn_theory_se, color=COLORS["blue"], linestyle="--", linewidth=1, label="Gaussian-score theory")
    axes[3].axhline(1 / (n - 1), color=COLORS["red"], linestyle="--", linewidth=1, label="chance")
    axes[3].set_title("D · Nearest-neighbor agreement")
    axes[3].set_ylabel("agreement probability")
    axes[3].legend()

    for method in PAPER_METHODS:
        mean = certificates["distortion_corr"][method].mean(axis=1)
        axes[4].plot(ratio_a, mean, color=COLORS_BY_METHOD[method], marker="o", label=LABELS[method])
    axes[4].axhline(0, color=COLORS["blue"], linestyle="--", linewidth=1)
    axes[4].plot(ratio_a, -np.sqrt(mgrid_a / (mgrid_a + d)), color=COLORS["red"], linestyle="--", linewidth=1, label="replacement, large-$d$")
    axes[4].set_title("E · Distortion vs. source distance")
    axes[4].set_ylabel(r"$\mathrm{Corr}(D_{\rm emb}/D-1,D)$")
    axes[4].legend()

    for method in PAPER_METHODS:
        axes[5].plot(ratio_a, shared[method], color=COLORS_BY_METHOD[method], marker="o", label=LABELS[method])
    axes[5].axhline(0.25, color=COLORS["dark_gray"], linestyle="--", linewidth=1, label="leading value 1/4")
    axes[5].set_title("F · Distortions sharing one point")
    axes[5].set_ylabel(r"$\mathrm{Corr}(e_{01},e_{02})$")
    axes[5].legend()

    for axis in axes:
        axis.set_xlabel(r"compression ratio  $m/d$")
        _format_ratio_axis(axis)
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def _write_certificate_csv(
    path: Path,
    mgrid: np.ndarray,
    d: int,
    certificates: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = [
        "method",
        "m",
        "d",
        "m_over_d",
        "trial",
        "max_error",
        "distance_corr",
        "distortion_corr",
        "calibrated_max_error",
        "calibrated_mean_error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            for index, m in enumerate(mgrid):
                for trial in range(certificates["max_error"][method].shape[1]):
                    writer.writerow(
                        {
                            "method": method,
                            "m": int(m),
                            "d": d,
                            "m_over_d": m / d,
                            "trial": trial,
                            "max_error": certificates["max_error"][method][index, trial],
                            "distance_corr": certificates["distance_corr"][method][index, trial],
                            "distortion_corr": certificates["distortion_corr"][method][index, trial],
                            "calibrated_max_error": (
                                certificates["calibrated_max_error"]["replacement"][
                                    index, trial
                                ]
                                if method == "replacement"
                                else ""
                            ),
                            "calibrated_mean_error": (
                                certificates["calibrated_mean_error"]["replacement"][
                                    index, trial
                                ]
                                if method == "replacement"
                                else ""
                            ),
                        }
                    )


def _write_ranking_csv(
    path: Path, mgrid: np.ndarray, d: int, rankings: dict[str, dict[str, np.ndarray]]
) -> None:
    fields = ["method", "m", "d", "m_over_d", "trial", "kendall", "nearest"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in PAPER_METHODS:
            for index, m in enumerate(mgrid):
                for trial in range(rankings["kendall"][method].shape[1]):
                    writer.writerow(
                        {
                            "method": method,
                            "m": int(m),
                            "d": d,
                            "m_over_d": m / d,
                            "trial": trial,
                            "kendall": rankings["kendall"][method][index, trial],
                            "nearest": rankings["nearest"][method][index, trial],
                        }
                    )


def _write_sharpness_csv(
    path: Path,
    *,
    mgrid: np.ndarray,
    d: int,
    epsilon: float,
    diagnostic: dict[str, np.ndarray],
) -> None:
    fields = [
        "m",
        "d",
        "m_over_d",
        "epsilon",
        "empirical_failure_probability",
        "exact_f_failure_probability",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, m in enumerate(mgrid):
            writer.writerow(
                {
                    "m": int(m),
                    "d": d,
                    "m_over_d": m / d,
                    "epsilon": epsilon,
                    "empirical_failure_probability": diagnostic["empirical"][index],
                    "exact_f_failure_probability": diagnostic["exact"][index],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "zero_information_jl")
    args = parser.parse_args()
    if not 0 < args.epsilon < 1:
        parser.error("epsilon must lie in (0,1)")
    config = PROFILES[args.profile]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    d, n = int(config["d"]), int(config["n"])
    mgrid_a = np.asarray(config["mgrid_a"], dtype=int)
    mgrid_c = np.asarray(config["mgrid_c"], dtype=int)

    certificates = simulate_certificates(
        rng, d=d, n=n, mgrid=mgrid_a, trials=int(config["trials_a"])
    )
    scatter = simulate_scatter_case(rng, d=d, n=n, m=int(config["m_b"]))
    rankings = simulate_rankings(
        rng, d=d, n=n, mgrid=mgrid_c, trials=int(config["trials_c"])
    )
    tau_theory = np.array([kendall_tau(int(m), d) for m in mgrid_c])
    nn_theory, nn_theory_se = plurality_kernel_mc_grid(
        rng,
        np.sqrt(mgrid_c / d),
        n - 1,
        int(config["kernel_reps"]),
    )
    shared = shared_pair_correlations(
        rng,
        d=d,
        mgrid=mgrid_a,
        repetitions=int(config["shared_reps"]),
    )
    sharpness = disjoint_pair_tail_diagnostic(
        rng,
        d=d,
        mgrid=mgrid_a,
        epsilon=args.epsilon,
        repetitions=int(config["sharpness_reps"]),
    )

    make_paper_figure(
        output / "paper_zero_information_jl",
        d=d,
        mgrid=mgrid_a,
        epsilon=args.epsilon,
        certificates=certificates,
        scatter=scatter,
        m_b=int(config["m_b"]),
    )
    make_dashboard(
        output / "dashboard_zero_information_jl",
        d=d,
        n=n,
        epsilon=args.epsilon,
        mgrid_a=mgrid_a,
        mgrid_c=mgrid_c,
        certificates=certificates,
        rankings=rankings,
        tau_theory=tau_theory,
        nn_theory=nn_theory,
        nn_theory_se=nn_theory_se,
        shared=shared,
    )
    _write_certificate_csv(output / "certificate_trials.csv", mgrid_a, d, certificates)
    _write_ranking_csv(output / "ranking_trials.csv", mgrid_c, d, rankings)
    _write_sharpness_csv(
        output / "sharpness.csv",
        mgrid=mgrid_a,
        d=d,
        epsilon=args.epsilon,
        diagnostic=sharpness,
    )
    np.savez_compressed(
        output / "scatter_case.npz",
        z_truth=scatter["z_truth"],
        z_haar=scatter["z_haar"],
        z_replacement=scatter["z_replacement"],
    )
    target = int(mgrid_a[np.argmin(np.abs(mgrid_a / d - 0.125))])
    index = int(np.flatnonzero(mgrid_a == target)[0])
    summary = {
        "schema_version": "1.0",
        "profile": args.profile,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "d": d,
        "n": n,
        "mgrid_certificate": mgrid_a.tolist(),
        "mgrid_ranking": mgrid_c.tolist(),
        "trials_certificate": int(config["trials_a"]),
        "trials_ranking": int(config["trials_c"]),
        "shared_pair_repetitions": int(config["shared_reps"]),
        "sharpness_repetitions": int(config["sharpness_reps"]),
        "scatter_m": int(config["m_b"]),
        "scatter_rho_haar": float(scatter["rho_haar"]),
        "scatter_rho_replacement": float(scatter["rho_replacement"]),
        "scatter_max_error_haar": float(scatter["max_error_haar"]),
        "scatter_max_error_replacement": float(scatter["max_error_replacement"]),
        "target_m": target,
        "target_calibrated_median_max_error": float(
            np.median(certificates["calibrated_max_error"]["replacement"][index])
        ),
        "maximum_calibrated_mean_error": float(
            certificates["calibrated_mean_error"]["replacement"].max()
        ),
        "maximum_f_tail_absolute_error": float(
            np.max(np.abs(sharpness["empirical"] - sharpness["exact"]))
        ),
        "haar_equivalence": "fixed-coordinate projection reproduces the joint law of the original and projected distance matrices for isotropic data",
        "shared_pair_estimator": "correlation across repeated independent triples",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote zero-information JL artifacts to {output}")
    print(f"m/d={target/d:.3f}; median max errors: " + ", ".join(
        f"{method}={np.median(certificates['max_error'][method][index]):.3f}" for method in METHODS
    ))
    print(
        f"scatter m/d={int(config['m_b'])/d:.3f}: rho Haar={float(scatter['rho_haar']):.3f}, "
        f"replacement={float(scatter['rho_replacement']):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
