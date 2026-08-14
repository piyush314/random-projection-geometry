"""Tiny harness for the machine-checked theorem suite.

Each ``verify_*.py`` module registers one or more checks with the
``@check`` decorator.  A check returns a list of ``(claim, ok, detail)``
triples.  ``run_all.py`` imports every module, runs the registry, prints a
table, writes ``results.md`` / ``results.json``, and exits nonzero on any
failure -- so CI goes red the moment a theorem's checkable consequence
stops holding.

Environment:
  RPGEOM_FULL=1   run at full (paper-scale) sample sizes instead of the
                  CI-friendly fast sizes.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

FULL = os.environ.get("RPGEOM_FULL", "0") == "1"

_REGISTRY: list[tuple[str, str, object]] = []


def check(theorem: str, description: str):
    def deco(fn):
        _REGISTRY.append((theorem, description, fn))
        return fn

    return deco


def run(out_dir: str | Path | None = None) -> int:
    out_dir = Path(out_dir or Path(__file__).parent)
    rows = []
    n_fail = 0
    mode = "FULL" if FULL else "FAST"
    print(f"rpgeom verification suite  [{mode} mode]")
    print("=" * 78)
    for theorem, desc, fn in _REGISTRY:
        t0 = time.time()
        try:
            results = fn()
            err = None
        except Exception:
            results = [("execution", False, traceback.format_exc(limit=3))]
            err = True
        dt = time.time() - t0
        ok_all = all(ok for _, ok, _ in results)
        n_fail += 0 if ok_all else 1
        status = "PASS" if ok_all else "FAIL"
        print(f"[{status}] {theorem:<12} {desc}  ({dt:.1f}s)")
        for claim, ok, detail in results:
            mark = "ok " if ok else "FAIL"
            print(f"       {mark} {claim}: {detail}")
        rows.append(
            {
                "theorem": theorem,
                "description": desc,
                "status": status,
                "seconds": round(dt, 1),
                "claims": [
                    {"claim": c, "ok": bool(o), "detail": d} for c, o, d in results
                ],
            }
        )
    print("=" * 78)
    print(f"{len(_REGISTRY) - n_fail}/{len(_REGISTRY)} checks passed  [{mode} mode]")

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    md = [
        "# Verification results",
        "",
        f"Mode: {mode}",
        "",
        "| Theorem | Check | Status | Time (s) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['theorem']} | {r['description']} | {r['status']} | {r['seconds']} |"
        )
    (out_dir / "results.md").write_text("\n".join(md) + "\n")
    return 0 if n_fail == 0 else 1


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol
