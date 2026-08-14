#!/usr/bin/env python3
"""Audit a real feature matrix against the exact Gaussian projection budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rpgeom import audit  # noqa: E402


def _load(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--q", type=int, default=10)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "audit.json")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    matrix = _load(args.input) if args.input else rng.standard_normal((1200, 128))
    if args.m >= matrix.shape[1]:
        parser.error(f"m must be below the feature dimension {matrix.shape[1]}")
    report = audit(matrix, args.m, args.q, args.trials, rng)
    payload = report.to_dict()
    metadata = {
        "schema_version": "1.0",
        "seed": args.seed,
        "source": str(args.input) if args.input else "synthetic-isotropic-smoke",
        "report_schema": "schemas/audit-report.schema.json",
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(report)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
