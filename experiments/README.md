# Experiments

Each experiment is a small, auditable unit with three files:

- `README.md` states the scientific question and expected interpretation.
- `contract.json` records inputs, outputs, defaults, and reproducibility promises.
- `run.py` is the executable reference implementation.

| Family | Question | Default runtime |
|---|---|---:|
| [`paper_tables`](paper_tables/) | Can every manuscript table and figure be regenerated? | minutes |
| [`finite_m`](finite_m/) | How quickly do asymptotic ranking laws become accurate? | seconds |
| [`anisotropic_hgr`](anisotropic_hgr/) | When can nonlinear recovery exceed the distance-value share? | seconds |
| [`zero_information_jl`](zero_information_jl/) | Can an independent cloud meet the same sampled JL inequalities? | seconds–minutes |
| [`realdata`](realdata/) | How closely does a supplied feature matrix follow the Gaussian budget? | seconds–minutes |

All stochastic runs take an explicit seed and write metadata beside their data.
Outputs under `artifacts/` are deliberately ignored by Git.
