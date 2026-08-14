import numpy as np

from rpgeom import ceilings, ensembles, mmse


def test_shape_ratio_table3():
    assert abs(ceilings.shape_ratio(8, 40) - 70 / 1638) < 1e-12


def test_alpha_m_top_heavy():
    lam = np.array([4.0, 1.0, 1.0, 1.0])
    assert ceilings.alpha_m(lam, 1) == 16 / 19


def test_laguerre_top_singval():
    assert abs(ceilings.laguerre_singvals(3, 7, 1)[0] - np.sqrt(3 / 7)) < 1e-12


def test_mutual_info_limits():
    assert abs(ceilings.mutual_info(12000, 40000) - (-0.5 * np.log(0.7))) < 2e-4


def test_chain_and_ensemble_constants():
    assert abs(ensembles.chain_corr([200], 200) - np.sqrt(200 / 402)) < 1e-12
    assert abs(ensembles.chain_corr([150, 100], 300) - 0.405767) < 1e-6
    assert abs(ensembles.averaging_corr(100, 400) - np.sqrt(100 / 502)) < 1e-12


def test_quarter_circle():
    assert abs(mmse.quarter_circle_loss(0.2) - 0.180998) < 1e-6
    assert abs(mmse.quarter_circle_loss(0.2) - mmse.quarter_circle_loss_quad(0.2)) < 1e-9
