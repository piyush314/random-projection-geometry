# Notebooks

The notebooks are executable interfaces to the project rather than screenshots
of one successful run.

- [`tutorials/`](tutorials/) contains the nine-lesson learning path mirrored by
  the Quarto site.
- [`reproduction/`](reproduction/) contains table, theorem-suite, and figure
  provenance walkthroughs.

Run everything from the repository root:

```bash
for notebook in notebooks/tutorials/*.ipynb notebooks/reproduction/*.ipynb; do
  jupyter nbconvert --to notebook --execute "$notebook" \
    --output-dir /tmp/rpgeom-notebooks --ExecutePreprocessor.timeout=600
done
```

CI executes clean copies and never commits transient outputs.
