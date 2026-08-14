# Documentation site

The Quarto source in this directory builds the project website, including the
browser calculator, nine-lesson learning path, theorem index, reproduction
guide, API contracts, FAQ, and citation page.

```bash
quarto preview docs
quarto render docs
```

GitHub Actions deploys `docs/_site` through the official Pages artifact flow.
The scientific Python runtime for the calculator loads client-side through
Pyodide; no server receives calculator inputs.
