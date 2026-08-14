import json
from pathlib import Path

from rpgeom import budget, recommend_dimension

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_valid_and_have_cell_ids():
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    assert len(notebooks) == 12
    for path in notebooks:
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        assert all("id" in cell for cell in notebook["cells"])


def test_experiment_contracts_cover_all_families():
    contracts = sorted((ROOT / "experiments").glob("*/contract.json"))
    assert {path.parent.name for path in contracts} == {
        "anisotropic_hgr",
        "finite_m",
        "paper_tables",
        "realdata",
    }
    for path in contracts:
        contract = json.loads(path.read_text())
        assert contract["schema_version"] == "1.0"
        assert contract["inputs"] and contract["outputs"] and contract["success"]


def test_report_keys_match_declared_schema_properties():
    budget_keys = set(budget(100, 10, q=5).to_dict())
    budget_schema = json.loads((ROOT / "schemas" / "budget-report.schema.json").read_text())
    assert budget_keys == set(budget_schema["properties"])

    recommendation_keys = set(recommend_dimension(100, kendall_tau=0.2).to_dict())
    recommendation_schema = json.loads(
        (ROOT / "schemas" / "dimension-recommendation.schema.json").read_text()
    )
    assert recommendation_keys == set(recommendation_schema["properties"])
