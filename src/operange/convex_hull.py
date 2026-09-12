"""An explicitly declared continuous convex hull of physical realizations."""

from dataclasses import dataclass
from functools import cached_property

from ._geometry import Geometry, check_tolerance, weights
from ._numeric import normalized_score
from .contract_types import Record, nonempty, unique
from .domains import DomainCapabilities, ParameterSpace, Scenario


@dataclass(frozen=True)
class ConvexHullSet(Record, Geometry):
    """All convex combinations of named generators, not just those scenarios.

    Generators need not be extreme or affinely independent. Their labels and
    provenance are retained, including duplicate realizations. Membership uses
    exact certificates; numerical candidates alone never enlarge the hull.
    """

    space: ParameterSpace
    vertices: tuple[Scenario, ...]
    provenance: str = "Caller-declared continuous convex hull"
    tolerance: float = 1e-10

    def _validate(self):
        if not self.vertices:
            raise ValueError("convex hull must have at least one generator")
        nonempty(self.provenance, "provenance")
        check_tolerance(self.tolerance)
        unique(tuple(s.name for s in self.vertices), "generator names")
        object.__setattr__(
            self, "vertices", tuple(sorted(self.vertices, key=lambda s: s.name))
        )
        for vertex in self.vertices:
            self.space.validate(vertex.values)

    @property
    def capabilities(self):
        return DomainCapabilities(
            linear_optimization=all(
                c.nominal is not None and c.scale is not None
                for c in self.space.coordinates
            )
        )

    @property
    def feasible_point(self):
        return self.vertices[0].values

    @cached_property
    def _bounds(self):
        return {
            name: (
                min(s.values[name] for s in self.vertices),
                max(s.values[name] for s in self.vertices),
            )
            for name in self.space.names
        }

    def to_manifest(self):
        return {
            "kind": "convex_hull",
            "space": self.space.to_dict(),
            "vertices": [s.to_dict() for s in self.vertices],
            "provenance": self.provenance,
            "tolerance": self.tolerance,
        }

    def membership(self, realization):
        from ._hull_membership import membership

        return membership(self, self.space.validate(realization))

    def maximize_linear(self, coefficients):
        w = weights(self.space, coefficients)
        if not self.capabilities.linear_optimization:
            return self._unresolved(
                w,
                "Hull support requires explicit nominals and scales.",
                status="unsupported",
            )
        try:
            scores = [normalized_score(self.space, w, s.values) for s in self.vertices]
            index = max(range(len(scores)), key=scores.__getitem__)
            return self._support(
                w,
                self.vertices[index].values,
                scores[index],
                "convex_hull_generator_maximum",
                details={
                    "generator_scores_exact": {
                        s.name: str(score) for s, score in zip(self.vertices, scores)
                    },
                    "formula": "a linear function over a convex hull attains its maximum at a generator",
                },
            )
        except (ValueError, OverflowError) as exc:
            return self._unresolved(w, str(exc))
