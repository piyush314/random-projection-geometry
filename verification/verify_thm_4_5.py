"""Theorem 4.5 (balanced spectral thinning): with equal per-block retention
theta, the full nonlinear ceiling is exactly sqrt(theta), attained by the
linear witness.  Exact-moment degree-2 CCA on the paper's two-block example
(lambda = 2,1; ranks 4,8; retained 1,2; theta = 1/4)."""

import sympy as sp

from harness import check


@check("Thm 4.5", "balanced thinning gives exactly sqrt(theta), exact moments")
def run():
    u1, v1, u2, v2 = sp.symbols("u1 v1 u2 v2", positive=True)
    dof = {u1: 1, v1: 3, u2: 2, v2: 6}

    def chi2_mom(k, p):
        out = sp.Integer(1)
        for j in range(p):
            out *= k + 2 * j
        return out

    def E(expr):
        expr = sp.expand(expr)
        tot = 0
        for term, coeff in expr.as_coefficients_dict().items():
            val = coeff
            for s in (u1, v1, u2, v2):
                p = sp.degree(term, s) if term.has(s) else 0
                val *= chi2_mom(dof[s], int(p))
            tot += val
        return sp.nsimplify(tot)

    lam1, lam2 = 2, 1
    D = lam1 * (u1 + v1) + lam2 * (u2 + v2)
    fx = [D - E(D), sp.expand(D**2) - E(D**2)]
    fy = [t - E(t) for t in [u1, u2, u1**2, u2**2, u1 * u2]]
    Sxx = sp.Matrix([[E(sp.expand(a * b)) for b in fx] for a in fx])
    Syy = sp.Matrix([[E(sp.expand(a * b)) for b in fy] for a in fy])
    Sxy = sp.Matrix([[E(sp.expand(a * b)) for b in fy] for a in fx])
    M = Sxx.inv() * Sxy * Syy.inv() * Sxy.T
    top = sorted([sp.nsimplify(e) for e in M.eigenvals()], key=lambda e: float(e))[-1]
    W = lam1 * u1 + lam2 * u2
    lin = E((D - E(D)) * (W - E(W))) / sp.sqrt(E((D - E(D)) ** 2) * E((W - E(W)) ** 2))
    return [
        (
            "max canonical rho^2 == theta = 1/4 exactly (symbolic)",
            sp.simplify(top - sp.Rational(1, 4)) == 0,
            f"max canonical corr = {float(sp.sqrt(top)):.9f} (claim: exactly 0.5)",
        ),
        (
            "linear witness attains sqrt(theta) (symbolic)",
            sp.simplify(lin - sp.Rational(1, 2)) == 0,
            f"linear witness = {float(lin):.9f}",
        ),
    ]
