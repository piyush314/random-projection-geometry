.PHONY: install test verify notebooks paper docs build quality

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

verify:
	python verification/run_all.py

notebooks:
	mkdir -p /tmp/rpgeom-notebooks
	jupyter nbconvert --to notebook --execute notebooks/tutorials/*.ipynb notebooks/reproduction/*.ipynb --output-dir /tmp/rpgeom-notebooks --ExecutePreprocessor.timeout=600

paper:
	python experiments/paper_tables/run.py --output artifacts/paper

docs:
	quarto render docs

build:
	python -m build

quality:
	ruff check .
	pytest -q
