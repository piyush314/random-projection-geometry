# Contributing

Thank you for helping make fine-geometry claims easier to understand and test.

## Good contributions

- a minimal failing example for a formula or input-validation edge case;
- a theorem check that is independent of the implementation under test;
- a clearer explanation, accessibility fix, or notebook correction;
- a real-data audit with redistributable data and a precise provenance note;
- an experiment contract that distinguishes theorem, approximation, and simulation.

## Local workflow

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python verification/run_all.py
```

Execute changed notebooks with `jupyter nbconvert --execute`; do not commit
transient outputs. Generated experiment artifacts belong under `artifacts/`.

## Pull requests

Keep a PR focused, explain the scientific meaning of the change, state any new
assumption, and include a test or verification update. Numerical changes should
record seeds, tolerances, and whether the reference is exact, asymptotic, or
Monte Carlo.
