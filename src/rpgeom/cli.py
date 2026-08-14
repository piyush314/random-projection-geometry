"""Command-line interface for budgets, audits, recommendations, and checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .audit import audit
from .budget import budget
from .recommend import recommend_dimension


def _load_array(path: str) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return np.load(source)
    delimiter = "," if source.suffix.lower() == ".csv" else None
    return np.loadtxt(source, delimiter=delimiter)


def _emit(report: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rpgeom",
        description="Compute and test fine-geometry budgets for random projections.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_budget = sub.add_parser("budget", help="compute the Gaussian-model budget")
    p_budget.add_argument("--d", type=int, required=True)
    p_budget.add_argument("--m", type=int, required=True)
    p_budget.add_argument("--q", type=int)
    p_budget.add_argument("--spectrum", help=".npy, .csv, or whitespace-delimited eigenvalues")
    p_budget.add_argument("--json", action="store_true")

    p_rec = sub.add_parser("recommend", help="find the smallest dimension meeting targets")
    p_rec.add_argument("--d", type=int, required=True)
    p_rec.add_argument("--q", type=int)
    p_rec.add_argument("--hgr-capacity", type=float)
    p_rec.add_argument("--pairwise-agreement", type=float)
    p_rec.add_argument("--kendall-tau", type=float)
    p_rec.add_argument("--nearest-neighbor", type=float)
    p_rec.add_argument("--shape-information", type=float)
    p_rec.add_argument("--json", action="store_true")

    p_audit = sub.add_parser("audit", help="compare theory with a data matrix")
    p_audit.add_argument("path", help="observations-by-features matrix (.npy, .csv, or text)")
    p_audit.add_argument("--m", type=int, required=True)
    p_audit.add_argument("--q", type=int, default=10)
    p_audit.add_argument("--trials", type=int, default=2000)
    p_audit.add_argument("--seed", type=int, default=0)
    p_audit.add_argument("--json", action="store_true")

    p_verify = sub.add_parser("verify", help="run the machine-checkable theorem suite")
    p_verify.add_argument("--full", action="store_true")

    p_repro = sub.add_parser("reproduce-paper", help="regenerate paper tables and figures")
    p_repro.add_argument("--full", action="store_true")
    p_repro.add_argument("--output", default="artifacts/paper")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "budget":
        spectrum = _load_array(args.spectrum).reshape(-1) if args.spectrum else None
        _emit(budget(args.d, args.m, args.q, spectrum), args.json)
        return 0
    if args.command == "recommend":
        report = recommend_dimension(
            args.d,
            q=args.q,
            hgr_capacity=args.hgr_capacity,
            pairwise_agreement=args.pairwise_agreement,
            kendall_tau=args.kendall_tau,
            nearest_neighbor=args.nearest_neighbor,
            shape_information=args.shape_information,
        )
        _emit(report, args.json)
        return 0
    if args.command == "audit":
        report = audit(
            _load_array(args.path),
            m=args.m,
            q=args.q,
            n_trials=args.trials,
            rng=np.random.default_rng(args.seed),
        )
        _emit(report, args.json)
        return 0
    if args.command == "verify":
        env = os.environ.copy()
        if args.full:
            env["RPGEOM_FULL"] = "1"
        return subprocess.call([sys.executable, "verification/run_all.py"], cwd=_repo_root(), env=env)
    if args.command == "reproduce-paper":
        command = [sys.executable, "experiments/paper_tables/run.py", "--output", args.output]
        if args.full:
            command.append("--full")
        return subprocess.call(command, cwd=_repo_root())
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
