import numpy as np

from rpgeom import audit, budget, recommend_dimension


def test_budget_report_runs():
    rep = budget(d=768, m=64, q=10, spectrum=np.linspace(1, 5, 768))
    s = str(rep)
    assert "HGR ceiling" in s and rep.nn_chance == 0.1
    assert 0 < rep.pairwise_agreement < 1
    assert rep.to_dict()["d"] == 768


def test_audit_matches_isotropic_gaussian():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1500, 200))
    rep = audit(X, m=20, q=8, n_trials=1500, rng=rng)
    # isotropic Gaussian data should track the exact laws
    assert abs(rep.meas_pairwise - rep.pred_pairwise) < 6 * rep.meas_pairwise_se + 0.01
    assert abs(rep.meas_nn - rep.pred_nn) < 6 * rep.meas_nn_se + 0.02
    assert rep.to_dict()["q"] == 8


def test_recommend_dimension_is_minimal():
    rep = recommend_dimension(100, kendall_tau=0.2)
    assert rep.achieved["kendall_tau"] >= 0.2
    if rep.m > 1:
        assert budget(100, rep.m - 1).kendall_tau < 0.2


def test_validation_rejects_bad_inputs():
    import pytest

    with pytest.raises(ValueError):
        budget(10, 10)
    with pytest.raises(ValueError):
        recommend_dimension(10)
