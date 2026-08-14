# Paper tables and figures

This is the archival reproduction entry point. It runs the original numerical
checks, the newer ranking and balanced-block checks, and the deterministic
figure generator. The default is a smoke-scale run; pass `--full` for the
sample sizes reported in the manuscript.

```bash
python experiments/paper_tables/run.py --output artifacts/paper
# archival scale
python experiments/paper_tables/run.py --full --output artifacts/paper
```

The command fails if a mathematical identity misses its declared tolerance.
