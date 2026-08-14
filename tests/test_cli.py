import json

from rpgeom.cli import main


def test_budget_json(capsys):
    assert main(["budget", "--d", "100", "--m", "10", "--q", "5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["d"] == 100
    assert payload["m"] == 10
    assert payload["nn_chance"] == 0.2


def test_recommend_json(capsys):
    assert main(["recommend", "--d", "100", "--kendall-tau", "0.2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["achieved"]["kendall_tau"] >= 0.2
