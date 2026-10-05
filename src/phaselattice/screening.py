"""Exact, conservative rejection of impossible waveforms from response ranges.

For g(t)=A(cos(2*pi*t))/Q, all extrema occur at the endpoints or
real derivative roots in [-1,1]. Rational root isolation and interval Horner
evaluation enclose those values. This is a necessary-condition filter only.
"""

from fractions import Fraction as F
from functools import lru_cache

import sympy as sp


def interval_polynomial(coefficients, lower, upper):
    lo = hi = F(0)
    for coefficient in coefficients:
        products = (lo * lower, lo * upper, hi * lower, hi * upper)
        lo, hi = min(products) + coefficient, max(products) + coefficient
    return lo, hi


@lru_cache(maxsize=4096)
def response_range(k, Q):
    z = sp.Symbol("z")
    polynomial = sp.Poly(sum(c * sp.chebyshevt(j + 1, z) for j, c in enumerate(k)), z)
    coefficients = tuple(F(c) for c in polynomial.all_coeffs())
    values = [interval_polynomial(coefficients, F(s), F(s)) for s in (-1, 1)]
    roots = []
    for (a, b), multiplicity in polynomial.diff().intervals(eps=sp.Rational(1, 2**64)):
        lo, hi = max(F(a), F(-1)), min(F(b), F(1))
        if lo <= hi:
            values.append(interval_polynomial(coefficients, lo, hi))
            roots.append(dict(interval=[str(lo), str(hi)], multiplicity=multiplicity))
    lower, upper = min(a for a, b in values) / Q, max(b for a, b in values) / Q
    return lower, upper, roots


def screen_candidates(public, data, family):
    # Training observations only; no fitted thresholds and no teacher access.
    labels = data["train"][1]
    observed_min, observed_max = min(labels), max(labels)
    # One dyadic unit is conservative for nearest rounding (half a unit).
    tolerance = F(1, 1 << public["B"])
    retained, rejected = [], []
    for k in sorted(family):
        lower, upper, roots = response_range(tuple(k), public["Q"])
        if observed_min < lower - tolerance or observed_max > upper + tolerance:
            rejected.append(
                dict(k=list(k), range=[str(lower), str(upper)], derivative_root_intervals=roots)
            )
        else:
            retained.append(k)
    return retained, dict(
        method="exact-rational-response-range",
        initial_candidates=len(family),
        retained_candidates=len(retained),
        observed_training_range=[str(observed_min), str(observed_max)],
        label_tolerance=str(tolerance),
        rejected=rejected,
    )
