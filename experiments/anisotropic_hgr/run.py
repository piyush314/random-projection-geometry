#!/usr/bin/env python3
"""Write exact-moment polynomial CCA lower bounds for anisotropic recovery."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reproduction.paper.verify_claims import anisotropy_sweep  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "anisotropic_hgr.csv"
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = anisotropy_sweep()
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    witness = min(rows, key=lambda row: abs(float(row["lambda_ratio"]) - 2.0))
    metadata = {
        "schema_version": "1.0",
        "rows": len(rows),
        "lambda_2_degree_2": witness["cca_degree_2"],
        "lambda_2_linear_ceiling": witness["sqrt_alpha_1"],
    }
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {len(rows)} exact-moment rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
