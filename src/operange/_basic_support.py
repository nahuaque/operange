"""Linear support for the existing box and finite domain declarations."""

from ._geometry import Geometry, weights
from ._numeric import normalized_score, round_up


class _SupportContext(Geometry):
    """Reuse numerical evidence conventions without changing domain manifests."""

    tolerance = 1e-10

    def __init__(self, domain):
        self.space = domain.space
        self.domain = domain
        self.feasible_point = domain.feasible_point

    @property
    def ref(self):
        return self.domain.ref

    def membership(self, realization):
        return self.domain.membership(realization)


def box_support(domain, coefficients):
    context = _SupportContext(domain)
    w = weights(domain.space, coefficients)
    try:
        point = {
            p.name: p.upper
            if w[p.name] > 0
            else p.lower
            if w[p.name] < 0
            else p.nominal
            for p in domain.scalar_parameters
        }
        upper = normalized_score(domain.space, w, point)
        return context._support(w, point, upper, "box_separable_extremum")
    except (ValueError, OverflowError) as exc:
        return context._unresolved(w, str(exc))


def finite_support(domain, coefficients):
    context = _SupportContext(domain)
    w = weights(domain.space, coefficients)
    if not domain.capabilities.linear_optimization:
        return context._unresolved(
            w,
            "Finite support requires explicit nominals and scales.",
            status="unsupported",
        )
    try:
        scores = [
            normalized_score(domain.space, w, s.values)
            for s in domain.scenarios
        ]
        index = max(range(len(scores)), key=scores.__getitem__)
        return context._support(
            w,
            domain.scenarios[index].values,
            scores[index],
            "finite_linear_enumeration",
            details={
                "scenario_scores": dict(
                    zip((s.name for s in domain.scenarios), map(round_up, scores))
                ),
                "maximizing_scenarios": [
                    s.name
                    for s, v in zip(domain.scenarios, scores)
                    if v == scores[index]
                ],
            },
        )
    except (ValueError, OverflowError) as exc:
        return context._unresolved(w, str(exc))
