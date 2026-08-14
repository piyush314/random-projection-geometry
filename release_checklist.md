# Release checklist

- [ ] ORNL/UT-Battelle software release clearance for this repository
      (separate from the manuscript's DOE publication notice; confirm the
      license text above matches the approved release terms)
- [ ] All verification checks green in CI (`python verification/run_all.py`)
- [ ] `pytest` green, smoke tests green (`RPGEOM_RUN_SMOKE=1 pytest -q`)
- [ ] All tutorial notebooks execute end-to-end through `nbconvert`
- [ ] Quarto renders with no broken internal links; Pages workflow is green
- [ ] Browser calculator checked in current Chrome, Firefox, and Safari
- [ ] README verified-claims table matches `verification/results.md`
- [ ] JSON outputs validate against all files under `schemas/`
- [ ] Wheel and source distribution install in clean Python 3.10 and 3.12 environments
- [ ] arXiv ID added to README + CITATION.cff once posted
- [ ] Change GitHub visibility from private to public only after clearance
- [ ] Enable GitHub Pages and PyPI trusted publishing environments
- [ ] Tag release; Zenodo GitHub integration mints software DOI; add DOI badge
- [ ] Journal DOI added after acceptance
