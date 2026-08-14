"""Run the full machine-checked theorem suite.

    python verification/run_all.py            # fast (CI) sizes
    RPGEOM_FULL=1 python verification/run_all.py   # paper-scale sizes
"""

import importlib
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import harness  # noqa: E402

for mod in sorted(HERE.glob("verify_*.py")):
    importlib.import_module(mod.stem)

if __name__ == "__main__":
    sys.exit(harness.run(HERE))
