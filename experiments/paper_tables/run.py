#!/usr/bin/env python3
"""Run all manuscript reproduction jobs and record an artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "paper")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    data = output / "data"
    figures = output / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    original = [
        sys.executable,
        "reproduction/paper/verify_claims.py",
        "--seed",
        "42",
        "--output-dir",
        str(data),
    ]
    newer = [
        sys.executable,
        "reproduction/paper/verify_new_results.py",
        "--seed",
        "20260812",
        "--output-dir",
        str(data),
    ]
    if args.full:
        original.append("--full")
        newer.append("--full")
    _run(original)
    _run(newer)
    _run([sys.executable, "scripts/generate_paper_assets.py", "--output", str(figures)])

    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full": args.full,
        "files": [
            {
                "path": str(path.relative_to(output)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in files
            if path.name != "manifest.json"
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest['files'])} artifacts and {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
