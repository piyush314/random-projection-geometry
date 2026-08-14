#!/usr/bin/env python3
"""Sweep finite sketch dimensions through the exact ranking laws."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rpgeom import laws  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=400)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "finite_m.csv")
    args = parser.parse_args()
    if args.d < 4 or args.q < 2:
        parser.error("need d >= 4 and q >= 2")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    grid = np.unique(np.geomspace(1, args.d - 1, 24).astype(int))
    rows = []
    for m in grid:
        exact = laws.pairwise_agreement(int(m), args.d)
        rows.append(
            {
                "d": args.d,
                "m": int(m),
                "m_over_d": m / args.d,
                "pairwise_exact": exact,
                "pairwise_small_ratio": 0.5 + np.sqrt(m / args.d) / np.pi,
                "kendall_exact": 2 * exact - 1,
                "nearest_of_q_limit": laws.plurality_kernel(np.sqrt(m / args.d), args.q),
                "chance": 1 / args.q,
            }
        )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"schema_version": "1.0", "d": args.d, "q": args.q, "rows": len(rows)}
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
