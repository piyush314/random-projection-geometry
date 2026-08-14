"""Heavier end-to-end smoke: RPGEOM_RUN_SMOKE=1 pytest tests/test_smoke_runs.py -q"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SMOKE = os.environ.get("RPGEOM_RUN_SMOKE", "0") == "1"


@pytest.mark.skipif(not SMOKE, reason="set RPGEOM_RUN_SMOKE=1 to run")
def test_full_verification_suite():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / "verification" / "run_all.py")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout[-2000:]
