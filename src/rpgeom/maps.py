"""Samplers for random maps: Haar frames, iid Gaussian stages, chains."""

from __future__ import annotations

import numpy as np

__all__ = ["haar_frame", "gaussian_stage", "gaussian_chain", "polar_split"]


def haar_frame(d: int, m: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """m x d matrix with orthonormal rows, Haar-distributed row space."""
    rng = np.random.default_rng() if rng is None else rng
    Q, _ = np.linalg.qr(rng.standard_normal((d, m)))
    return Q.T


def gaussian_stage(n_out: int, n_in: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """n_out x n_in iid Gaussian stage with entries N(0, 1/n_out) (SM1.1 convention)."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.standard_normal((n_out, n_in)) / np.sqrt(n_out)


def gaussian_chain(widths: list[int], d: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Product P_L = G_L ... G_1 of independent Gaussian stages applied to R^d."""
    rng = np.random.default_rng() if rng is None else rng
    P = np.eye(d)
    n_in = d
    for w in widths:
        P = gaussian_stage(w, n_in, rng) @ P
        n_in = w
    return P


def polar_split(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor an m x d Gaussian stage as (Wishart factor, Haar partial isometry).

    G = W^{1/2} R with W = G G^T and R = W^{-1/2} G having orthonormal rows.
    The Haar factor carries the rank bottleneck; the Wishart factor adds the
    multiplicative norm noise (proof of Theorem SM1.1 / SM2.10).
    """
    W = G @ G.T
    vals, vecs = np.linalg.eigh(W)
    W_half = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    W_ihalf = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return W_half, W_ihalf @ G
