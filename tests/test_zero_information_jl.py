import numpy as np

from experiments.zero_information_jl.run import (
    calibrate_replacement_distances,
    disjoint_pair_tail_diagnostic,
    shared_pair_correlations,
)


def test_empirical_calibration_matches_the_realized_baseline_and_preserves_ranks():
    original = np.array([8.0, 5.0, 11.0, 7.0])
    replacement = np.array([3.0, 9.0, 4.0, 6.0])
    calibrated = calibrate_replacement_distances(original, replacement)

    assert np.isclose(calibrated.mean(), original.mean())
    assert np.array_equal(np.argsort(calibrated), np.argsort(replacement))


def test_disjoint_pair_failure_matches_the_exact_f_tail():
    diagnostic = disjoint_pair_tail_diagnostic(
        np.random.default_rng(11),
        d=256,
        mgrid=np.array([32, 64, 128]),
        epsilon=0.2,
        repetitions=50_000,
    )

    assert np.max(np.abs(diagnostic["empirical"] - diagnostic["exact"])) < 0.012


def test_shared_pair_correlation_is_estimated_across_repetitions():
    correlations = shared_pair_correlations(
        np.random.default_rng(7),
        d=128,
        mgrid=np.array([16, 32]),
        repetitions=2500,
        batch_size=250,
    )
    for values in correlations.values():
        assert np.all((0.15 < values) & (values < 0.35))
        assert not np.allclose(values, -1 / 48)
