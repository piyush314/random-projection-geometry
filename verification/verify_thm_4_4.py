"""Theorem 8.4 (anisotropic nonlinear excess): for D = 2 G1^2 + G2^2 observed
through U = 2 G1^2, quadratic features achieve canonical rho^2 =
(246 + 2 sqrt(201))/311 > 4/5.  Verified *symbolically* -- exact equality of
radicals, not numerics.  The filename retains the number from the earlier
manuscript layout."""

import sympy as sp

from harness import check


@check("Thm 8.4", "nonlinear excess constant, symbolic equality")
def run():
    mom = {0: 1, 1: 1, 2: 3, 3: 15, 4: 105}
    A, B = sp.symbols("A B", positive=True)

    def E(expr):
        expr = sp.expand(expr)
        total = 0
        for term, coeff in expr.as_coefficients_dict().items():
            pa = sp.degree(term, A) if term.has(A) else 0
            pb = sp.degree(term, B) if term.has(B) else 0
            total += coeff * mom[int(pa)] * mom[int(pb)]
        return sp.nsimplify(total)

    D = 2 * A + B
    fx = [D - E(D), sp.expand(D**2 - E(D**2))]
    fy = [A - E(A), sp.expand(A**2 - E(A**2))]
    gram = lambda f1, f2: sp.Matrix([[E(sp.expand(a * b)) for b in f2] for a in f1])
    Sxx, Syy, Sxy = gram(fx, fx), gram(fy, fy), gram(fx, fy)
    M = Sxx.inv() * Sxy * Syy.inv() * Sxy.T
    rho2 = sorted(
        [sp.nsimplify(sp.radsimp(e)) for e in M.eigenvals()], key=lambda e: float(e)
    )[-1]
    claimed = sp.Rational(246, 311) + 2 * sp.sqrt(201) / 311
    match = sp.simplify(rho2 - claimed) == 0
    lin = E((D - E(D)) * (A - 1)) / sp.sqrt(E((D - E(D)) ** 2) * E((A - 1) ** 2))
    lin_ok = sp.simplify(lin**2 - sp.Rational(4, 5)) == 0
    return [
        (
            "quadratic CCA eigenvalue == (246+2*sqrt(201))/311 (symbolic)",
            bool(match),
            f"rho = {float(sp.sqrt(rho2)):.9f} > sqrt(4/5) = {float(sp.sqrt(sp.Rational(4,5))):.9f}",
        ),
        (
            "linear witness == sqrt(alpha) = sqrt(4/5) (symbolic)",
            bool(lin_ok),
            f"linear corr = {float(lin):.9f}",
        ),
    ]
