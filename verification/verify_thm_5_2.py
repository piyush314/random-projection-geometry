"""Theorem 9.4 (JL and ranking collapse coexist): a sampled map satisfies a
JL guarantee on all pairs while mean Kendall tau is near zero.  Direct
demonstration at log n << m << d, averaged over independent replicas so the
collapse *level* can be compared against the exact law (a single replica
only concentrates at the Hoeffding rate exp(-n t^2 / 6), which is vacuous
at these n).  The filename retains the number from the earlier manuscript
layout."""

import numpy as np
from scipy import stats as st
from scipy.spatial.distance import squareform, pdist

from harness import FULL, check
from rpgeom.laws import kendall_tau


@check("Thm 9.4", "one map: JL holds on all pairs AND rankings collapse")
def run():
    rng = np.random.default_rng(3)
    if FULL:
        n, d, m, reps = 100, 100_000, 400, 12
    else:
        n, d, m, reps = 60, 20_000, 200, 10
    eps_max = 0.0
    tau_reps = []
    for _ in range(reps):
        X = rng.standard_normal((n, d))
        Q, _ = np.linalg.qr(rng.standard_normal((d, m)))
        Y = (X @ Q) * np.sqrt(d / m)
        Dfull = squareform(pdist(X, "sqeuclidean"))
        Dproj = squareform(pdist(Y, "sqeuclidean"))
        iu = np.triu_indices(n, 1)
        eps_max = max(eps_max, float(np.abs(Dproj[iu] / Dfull[iu] - 1.0).max()))
        taus = []
        for i in range(n):
            others = np.delete(np.arange(n), i)
            taus.append(st.kendalltau(Dfull[i, others], Dproj[i, others]).statistic)
        tau_reps.append(float(np.mean(taus)))
    tau_bar = float(np.mean(tau_reps))
    tau_se = float(np.std(tau_reps) / np.sqrt(reps))
    theory = kendall_tau(m, d)
    return [
        (
            f"JL certificate holds on every pair, all {reps} replicas",
            eps_max < 0.5,
            f"max relative distortion {eps_max:.3f} (JL holds at eps = 0.5)",
        ),
        (
            "rankings simultaneously collapse",
            abs(tau_bar) < 0.12,
            f"mean Kendall tau {tau_bar:.4f} +- {tau_se:.4f} (perfect ranking = 1.0)",
        ),
        (
            "collapse level tracks the exact law",
            abs(tau_bar - theory) < max(0.03, 5 * tau_se),
            f"replica mean {tau_bar:.4f} vs exact {theory:.4f}",
        ),
    ]
