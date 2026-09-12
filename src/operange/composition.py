"""Typed domain composition with explicit coordinate alignment and witnesses."""

from dataclasses import dataclass
from typing import ClassVar

from ._geometry import Geometry, check_tolerance, weights
from .contract_types import Evidence, freeze, nonempty
from .domains import DomainCapabilities, MembershipCheck, ParameterSpace, UncertaintySet


def _merge_space(factors, explicit, kind):
    by_name = {}
    for factor in factors:
        for coordinate in factor.space.coordinates:
            previous = by_name.get(coordinate.name)
            if previous is not None:
                if kind == "product":
                    raise ValueError(
                        "product factors must have disjoint coordinate names"
                    )
                if not previous.aligned_with(coordinate):
                    raise ValueError("shared coordinate units or axes conflict")
                if explicit is None and previous != coordinate:
                    raise ValueError(
                        "shared coordinate definitions differ; supply an explicit joint space"
                    )
            by_name[coordinate.name] = coordinate
    if kind == "union" and any(
        set(f.space.names) != set(factors[0].space.names) for f in factors
    ):
        raise ValueError("union factors must cover the same coordinate names")
    merged = ParameterSpace(tuple(by_name.values()))
    if explicit is None:
        return merged
    if not isinstance(explicit, ParameterSpace):
        raise ValueError("space must be a ParameterSpace")
    explicit.check_alignment(merged)
    return explicit


class _Composition(Geometry):
    kind: ClassVar[str]

    def __post_init__(self):
        object.__setattr__(self, "factors", tuple(self.factors))
        if not self.factors or not all(
            isinstance(f, UncertaintySet) for f in self.factors
        ):
            raise ValueError("composition requires nonempty uncertainty-set factors")
        nonempty(self.provenance, "provenance")
        check_tolerance(self.tolerance)
        object.__setattr__(
            self, "space", _merge_space(self.factors, self.space, self.kind)
        )
        candidates = []
        if self.feasible_point is not None:
            candidates.append(self.space.validate(self.feasible_point))
        else:
            try:
                candidates.append(self.space.nominal)
            except ValueError:
                pass
            if self.kind == "product":
                point = {}
                for factor in self.factors:
                    member = getattr(factor, "feasible_point", None)
                    if member is None:
                        break
                    point.update(member)
                if set(point) == set(self.space.names):
                    candidates.append(point)
            else:
                for factor in self.factors:
                    member = getattr(factor, "feasible_point", None)
                    if member is not None and set(member) == set(self.space.names):
                        candidates.append(dict(member))
        for candidate in candidates:
            object.__setattr__(self, "feasible_point", freeze(candidate))
            if self.membership(candidate).status == "inside":
                return
        raise ValueError(
            "composition needs a verified feasible_point; empty or unresolved domains cannot certify a claim"
        )

    @property
    def capabilities(self):
        return DomainCapabilities()

    def to_manifest(self):
        return {
            "kind": self.kind,
            "space": self.space.to_dict(),
            "factors": [f.to_manifest() for f in self.factors],
            "feasible_point": dict(self.feasible_point)
            if self.feasible_point is not None
            else None,
            "provenance": self.provenance,
            "tolerance": self.tolerance,
        }

    def membership(self, realization):
        values = self.space.validate(realization)
        checks = [
            f.membership({n: values[n] for n in f.space.names}) for f in self.factors
        ]
        states = [c.status for c in checks]
        if self.kind == "union":
            status = (
                "inside"
                if "inside" in states
                else "outside"
                if all(s == "outside" for s in states)
                else "unknown"
            )
        else:
            status = (
                "outside"
                if "outside" in states
                else "inside"
                if all(s == "inside" for s in states)
                else "unknown"
            )
        return MembershipCheck(
            status,
            self.ref,
            values,
            (
                Evidence(
                    "membership",
                    "domain_membership",
                    f"{self.kind}_membership",
                    "unresolved" if status == "unknown" else "verified",
                    details={"factor_checks": [c.to_dict() for c in checks]},
                ),
            ),
        )

    def maximize_linear(self, coefficients):
        return self._unresolved(
            weights(self.space, coefficients),
            "This composition currently provides membership only; no general support-function solver is advertised.",
            status="unsupported",
        )


@dataclass(frozen=True)
class Intersection(_Composition):
    """Conjoin restrictions, lifting each factor onto its named coordinates."""

    factors: tuple[UncertaintySet, ...]
    space: ParameterSpace | None = None
    feasible_point: object = None
    provenance: str = "Caller-declared intersection"
    tolerance: float = 1e-10
    backend: str | None = None
    kind: ClassVar[str] = "intersection"

    def __post_init__(self):
        if self.backend not in (None, "cvxpy"):
            raise ValueError("intersection backend must be None or 'cvxpy'")
        super().__post_init__()

    @property
    def capabilities(self):
        if self.backend == "cvxpy":
            from ._intersection_support import supported

            return DomainCapabilities(linear_optimization=supported(self))
        return DomainCapabilities()

    def to_manifest(self):
        manifest = super().to_manifest()
        if self.backend is not None:
            manifest["backend"] = self.backend
        return manifest

    def maximize_linear(self, coefficients):
        if self.backend == "cvxpy":
            from ._intersection_support import maximize

            return maximize(self, weights(self.space, coefficients))
        return super().maximize_linear(coefficients)


@dataclass(frozen=True)
class Union(_Composition):
    """Admit any factor over the same physical coordinates; no convexification."""

    factors: tuple[UncertaintySet, ...]
    space: ParameterSpace | None = None
    feasible_point: object = None
    provenance: str = "Caller-declared union"
    tolerance: float = 1e-10
    kind: ClassVar[str] = "union"


@dataclass(frozen=True)
class Product(_Composition):
    """Combine disjoint coordinate spaces; no probability independence claim."""

    factors: tuple[UncertaintySet, ...]
    space: ParameterSpace | None = None
    feasible_point: object = None
    provenance: str = "Caller-declared Cartesian product"
    tolerance: float = 1e-10
    kind: ClassVar[str] = "product"
