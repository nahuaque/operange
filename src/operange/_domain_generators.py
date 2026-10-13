"""Deterministic traversal only; callers own the theorem giving it coverage."""

from itertools import product

from .convex_hull import ConvexHullSet
from .domains import FiniteSet, Scenario
from .primitives import BoxSet


def generator_count(domain):
    if type(domain) is BoxSet:
        return 2 ** sum(p.lower != p.upper for p in domain.scalar_parameters)
    if type(domain) is ConvexHullSet:
        return len(domain.vertices)
    if type(domain) is FiniteSet:
        return len(domain.scenarios)
    raise NotImplementedError(
        "Generators require a built-in box, finite set or explicit convex hull."
    )


def generators(domain, *, provenance):
    """Yield complete generators, with constant box coordinates appearing once.

    Count and enforce the study's budget before calling this iterator. A finite
    collection alone establishes no continuous-domain or policy guarantee.
    """
    if type(domain) is FiniteSet:
        yield from domain.scenarios
    elif type(domain) is ConvexHullSet:
        yield from domain.vertices
    elif type(domain) is BoxSet:
        parameters = sorted(domain.scalar_parameters, key=lambda p: p.name)
        for i, values in enumerate(
            product(
                *(
                    (p.lower,) if p.lower == p.upper else (p.lower, p.upper)
                    for p in parameters
                )
            )
        ):
            yield Scenario(
                f"corner_{i}",
                dict(zip((p.name for p in parameters), values)),
                provenance,
            )
    else:
        raise NotImplementedError(
            "Generators require a built-in box, finite set or explicit convex hull."
        )
