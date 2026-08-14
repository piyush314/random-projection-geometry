import numpy as np
import pytest

from rpgeom import laws


def test_pairwise_agreement_table3():
    assert abs(laws.pairwise_agreement(5, 50) - 0.59829) < 5e-6


def test_kendall_tau_table4():
    assert abs(laws.kendall_tau(2000, 1_000_000) - 0.028476) < 5e-6


def test_plurality_kernel_paper_value():
    assert abs(laws.plurality_kernel(np.sqrt(20 / 10_000), 8) - 0.138324998) < 2e-6


def test_plurality_kernel_sheppard_q2():
    for rho in (0.1, 0.4386, 0.8):
        exact = 0.5 + np.arcsin(rho) / np.pi
        assert abs(laws.plurality_kernel(rho, 2) - exact) < 2e-6


def test_plurality_kernel_limits():
    assert laws.plurality_kernel(0.0, 5) == pytest.approx(0.2)
    assert laws.plurality_kernel(1.0, 5) == pytest.approx(1.0)


def test_slope_c2_is_one_over_pi():
    assert abs(laws.slope_cq(2) - 1 / np.pi) < 1e-9


def test_coupling_delta_vanishes():
    # Delta decays like d^{-1/2} at fixed (q, m)
    assert laws.coupling_delta(3, 2, 1_000_000) < 0.05
    assert laws.coupling_delta(3, 2, 100_000_000) < 0.005
    assert laws.coupling_delta(8, 20, 10_000) == 1.0  # honest vacuity at Table-4 sizes
