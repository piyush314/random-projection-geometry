"""Theorem SM1.1: exact Gaussian-chain identity
Corr^2 = (2/d)/(prod_{l>=0}(1+2/n_l) - 1) at every finite size, checked by
the chi-square stage representation and against the paper's Table 3 chains."""

import numpy as np

from harness import FULL, check
from rpgeom.ensembles import chain_corr


@check("Thm SM1.1", "Gaussian-chain identity: closed form + stage-factor MC")
def run():
    rng = np.random.default_rng(8)
    out = []
    c1 = chain_corr([200], 200)
    out.append(
        ("one stage 200->200 (Table 3)", abs(c1 - 0.70535) < 5e-5, f"{c1:.5f} vs 0.70535")
    )
    c2 = chain_corr([150, 100], 300)
    out.append(
        ("chain 300->150->100 (Table 3)", abs(c2 - 0.40577) < 5e-5, f"{c2:.5f} vs 0.4058")
    )
    N = 2_000_000 if FULL else 800_000
    d, widths = 300, [150, 100]
    D = rng.chisquare(d, N)
    Q = D.copy()
    for w in widths:
        Q = Q * rng.chisquare(w, N) / w
    mc = np.corrcoef(D, Q)[0, 1]
    out.append(
        (
            "stage-representation MC matches the identity",
            abs(mc - c2) < 0.005,
            f"MC {mc:.5f} vs exact {c2:.5f}",
        )
    )
    sq = chain_corr([1000], 1000)
    out.append(
        (
            "square Gaussian stage -> 1/sqrt(2)",
            abs(sq - np.sqrt(1000 / 2002)) < 1e-12,
            f"{sq:.6f} (limit {1/np.sqrt(2):.6f})",
        )
    )
    return out
