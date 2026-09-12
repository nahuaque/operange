"""Support identities for products and unions, with exact coordinate transforms.

Products sum factor bounds; unions take their maximum. A union witness always
comes from one actual factor. No joint optimizer or convexification is needed.
"""

from fractions import Fraction as F

from ._geometry import LinearSupport, require_normalization
from ._numeric import normalized_score, round_down
from .primitives import finite


def supported(domain):
    try:
        require_normalization(domain.space)
        for factor in domain.factors:
            require_normalization(factor.space)
            if not factor.capabilities.linear_optimization or not callable(
                getattr(factor, "maximize_linear", None)
            ):
                return False
        return True
    except (ValueError, OverflowError):
        return False


def _checked_support(factor, coefficients):
    support = factor.maximize_linear(coefficients)
    if (
        not isinstance(support, LinearSupport)
        or support.domain_ref != factor.ref
        or dict(support.coefficients) != coefficients
    ):
        raise ValueError("factor support does not match its domain and objective")
    if support.status not in ("optimal", "bounded"):
        raise ValueError(f"factor support is {support.status}")
    point = factor.space.validate(support.point)
    check = factor.membership(point)
    if (
        check.status != "inside"
        or check.domain_ref != factor.ref
        or dict(check.realization) != point
    ):
        raise ValueError("factor support witness could not be verified")
    score = normalized_score(factor.space, coefficients, point)
    if (
        F(support.lower) > score
        or abs(round_down(score) - support.lower) > support.tolerance
        or F(support.upper) < score
    ):
        raise ValueError("factor support bounds disagree with its witness")
    return support


def _factor_support(space, factor, weights):
    exact, shift = {}, F(0)
    for local in factor.space.coordinates:
        joint = space.coordinate(local.name)
        exact[local.name] = F(weights[local.name]) * F(local.scale) / F(joint.scale)
        shift += (
            F(weights[local.name])
            * (F(local.nominal) - F(joint.nominal))
            / F(joint.scale)
        )
    rounded = {
        n: finite(float(v), "transformed support coefficient") for n, v in exact.items()
    }
    support = _checked_support(factor, rounded)
    residual = {n: v - F(rounded[n]) for n, v in exact.items()}
    correction, corrections = F(0), []
    # A rounding error in a transformed coefficient is bounded using support
    # in that coordinate's signed direction. This also covers float underflow.
    for name, value in residual.items():
        if value == 0:
            continue
        direction = {
            n: (1.0 if value > 0 else -1.0) if n == name else 0.0
            for n in factor.space.names
        }
        axis = _checked_support(factor, direction)
        term = abs(value) * F(axis.upper)
        correction += term
        corrections.append(
            {
                "coordinate": name,
                "residual_exact": str(value),
                "support": axis.to_dict(),
                "correction_exact": str(term),
            }
        )
    upper = shift + F(support.upper) + correction
    return (
        support.point,
        upper,
        {
            "factor_ref": factor.ref.to_dict(),
            "support": support.to_dict(),
            "coefficients_exact": {n: str(v) for n, v in exact.items()},
            "shift_exact": str(shift),
            "coefficient_corrections": corrections,
            "correction_exact": str(correction),
            "upper_exact": str(upper),
        },
    )


def maximize(domain, weights):
    if not supported(domain):
        return domain._unresolved(
            weights,
            "Composition support requires normalized spaces and a linear support operation for every factor.",
            status="unsupported",
        )
    records, points, uppers = [], [], []
    for index, factor in enumerate(domain.factors):
        try:
            point, upper, record = _factor_support(domain.space, factor, weights)
        except (ValueError, OverflowError) as exc:
            return domain._unresolved(weights, f"Factor {index}: {exc}")
        records.append({"factor_index": index, **record})
        points.append(point)
        uppers.append(upper)
    if domain.kind == "product":
        point = {name: value for part in points for name, value in part.items()}
        upper = sum(uppers, F(0))
        witness_factors = list(range(len(points)))
    else:
        index = max(
            range(len(points)),
            key=lambda i: normalized_score(domain.space, weights, points[i]),
        )
        point, upper = points[index], max(uppers)
        witness_factors = [index]
    try:
        if normalized_score(domain.space, weights, point) > upper:
            raise ValueError("composed support bound disagrees with its witness")
        return domain._support(
            weights,
            point,
            upper,
            f"{domain.kind}_support_identity",
            details={
                "formula": "sum(factor_supports)"
                if domain.kind == "product"
                else "max(factor_supports)",
                "factors": records,
                "witness_factor_indices": witness_factors,
                "upper_exact": str(upper),
                "arithmetic": "exact_rationals_of_declared_floats; bounds rounded outward",
                "scope": "all factors; normalized objectives transformed by name; witnesses retain original membership",
            },
        )
    except (ValueError, OverflowError) as exc:
        return domain._unresolved(weights, str(exc))
