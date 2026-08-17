# Paper tables and figures

This is the archival reproduction entry point. It runs the original numerical
checks, the newer ranking and balanced-block checks, and the deterministic
figure generator. It also runs the seeded zero-information JL experiment and
copies its trimmed publication figure into the paper bundle. The bundle contains
the five current manuscript plots plus the legacy Laguerre plot; the manuscript
builds its channel diagram separately from DOT. The default is a smoke-scale
run; pass `--full` for the sample sizes reported in the manuscript.

```bash
python experiments/paper_tables/run.py --output artifacts/paper
# archival scale
python experiments/paper_tables/run.py --full --output artifacts/paper
```

The command fails if a mathematical identity misses its declared tolerance.
