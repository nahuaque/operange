"""Common numerical conventions for uncertainty geometry operations."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Literal

from .contract_types import (
    ArtifactRef,
    Evidence,
    Measurement,
    Record,
    reference,
    unique,
)
from .domains import DomainCapabilities, MembershipCheck
from .primitives import finite
from ._numeric import normalized_score, round_down, round_up


def weights(space, coefficients):
    if not isinstance(coefficients, Mapping) or any(
        name not in space.names for name in coefficients
    ):
        raise ValueError("linear coefficients must name declared coordinates")
    return {name: finite(coefficients.get(name, 0), name) for name in space.names}


def dot(a, b):
    return finite(fsum(x * y for x, y in zip(a, b)), "linear expression")


def normalized_bounds(box):
    return tuple(
        ((p.lower - p.nominal) / p.scale, (p.upper - p.nominal) / p.scale)
        for p in box.scalar_parameters
    )


def require_normalization(space):
    space.normalize(space.nominal)


def check_tolerance(tolerance):
    if not 1e-10 <= tolerance <= 1e-4:
        raise ValueError("numerical tolerance must be between 1e-10 and 1e-4")


def classification(residuals, tolerance):
    """Do not widen the set: small positive residuals remain unknown."""
    if any(not isfinite(r) for r in residuals):
        return "unknown"
    if any(r > tolerance for r in residuals):
        return "outside"
    return "inside" if all(r <= 0 for r in residuals) else "unknown"


def membership_record(domain, values, residuals, method, *, details=None):
    status = classification(residuals, domain.tolerance)
    measurements = tuple(
        Measurement(f"residual_{i}", r, "1")
        for i, r in enumerate(residuals)
        if isfinite(r)
    )
    return MembershipCheck(
        status,
        domain.ref,
        values,
        (
            Evidence(
                "membership",
                "domain_membership",
                method,
                "unresolved" if status == "unknown" else "verified",
                measurements,
                (Measurement("classification_band", domain.tolerance, "1"),),
                details=details or {},
            ),
        ),
    )


@dataclass(frozen=True)
class LinearSupport(Record):
    """Bounds on max sum_i coefficients[i] * normalized_coordinate[i].

    This is a domain operation, not a process-robustness certificate. All scores
    are dimensionless; the returned witness uses physical coordinate units.
    """

    status: Literal["optimal", "bounded", "unresolved", "unsupported"]
    domain_ref: ArtifactRef
    coefficients: Mapping[str, float]
    point: Mapping[str, float] | None
    lower: float | None
    upper: float | None
    tolerance: float
    evidence: tuple[Evidence, ...]

    def _validate(self):
        check_tolerance(self.tolerance)
        unique(tuple(e.evidence_id for e in self.evidence), "support evidence")
        for name, value in self.coefficients.items():
            finite(value, name)
        if self.point is not None:
            for name, value in self.point.items():
                finite(value, name)
        if (self.lower is None) != (self.point is None):
            raise ValueError("a support lower bound requires its feasible witness")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ValueError("support bounds are reversed")
        if self.status == "optimal" and (
            self.lower is None
            or self.upper is None
            or self.upper - self.lower > self.tolerance
        ):
            raise ValueError("optimal support requires closed bounds")
        if self.status == "bounded" and (self.lower is None or self.upper is None):
            raise ValueError("bounded support requires both bounds")
        if self.status in ("unsupported", "unresolved") and (
            self.point is not None or self.lower is not None or self.upper is not None
        ):
            raise ValueError("unresolved support cannot fabricate bounds")
        if self.lower is not None and not any(
            e.subject == "domain_membership" and e.outcome == "verified"
            for e in self.evidence
        ):
            raise ValueError("support witness needs membership evidence")
        if self.upper is not None and not any(
            e.subject == "support_upper_bound" and e.outcome == "verified"
            for e in self.evidence
        ):
            raise ValueError("support upper bound needs evidence")


class Geometry:
    @property
    def ref(self):
        return reference(self.to_manifest(), "uncertainty_set/v1")

    @property
    def capabilities(self):
        return DomainCapabilities(linear_optimization=True)

    def contains(self, realization):
        status = self.membership(realization).status
        if status == "unknown":
            raise ValueError(
                "membership is numerically unresolved; inspect membership()"
            )
        return status == "inside"

    def _support(self, coefficients, point, upper, method, *, details=None, extra=()):
        w = weights(self.space, coefficients)
        check = self.membership(point)
        if check.status != "inside":
            point = self.feasible_point
            check = self.membership(point)
        if check.status != "inside":
            return self._unresolved(w, "could not verify a feasible support witness")
        lower = round_down(normalized_score(self.space, w, point))
        upper = round_up(upper)
        if lower > upper + self.tolerance:
            return self._unresolved(
                w, "analytical/dual bound disagrees with feasible point"
            )
        upper = max(lower, upper)
        evidence = (
            *check.evidence,
            Evidence(
                "upper",
                "support_upper_bound",
                method,
                "verified",
                (Measurement("upper", upper, "1"),),
                (Measurement("absolute_tolerance", self.tolerance, "1"),),
                details=details or {},
            ),
            *extra,
        )
        return LinearSupport(
            "optimal" if upper - lower <= self.tolerance else "bounded",
            self.ref,
            w,
            point,
            lower,
            upper,
            self.tolerance,
            evidence,
        )

    def _unresolved(self, coefficients, message, *, status="unresolved"):
        return LinearSupport(
            status,
            self.ref,
            coefficients,
            None,
            None,
            None,
            self.tolerance,
            (
                Evidence(
                    "support",
                    "linear_support",
                    "unresolved_calculation",
                    "unresolved",
                    details={"message": message},
                ),
            ),
        )

    def _interior_candidate(self, point, center):
        """Move an analytical boundary candidate slightly inward after rounding."""
        if self.membership(point).status == "inside":
            return point
        for factor in (1 - 1e-14, 1 - 1e-12, 1 - 1e-10):
            candidate = {
                n: center[n] + factor * (point[n] - center[n]) for n in self.space.names
            }
            if self.membership(candidate).status == "inside":
                return candidate
        return center
