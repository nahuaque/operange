"""Typed study intent and request-specific, solver-free capability reports.

Requests bind to an existing Claim (including its operating policy), or to an
IDAES adapter for studies whose domain/policy is part of the request. They are
runtime inputs, not a new artifact format. Results retain their family semantics.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import ClassVar, Literal

from .contract_types import Record
from .dispatch_sensitivity import DispatchChange
from .domains import UncertaintySet
from .objectives import ControlTrackingObjective, LinearObjective
from .operating import OperatingSpec
from .parameters import ParameterCovariance


EvidenceRequirement = Literal[
    "checked_feasibility",
    "infeasibility_certificate",
    "objective_bounds",
    "continuous_coverage",
    "controller_audit",
    "local_sensitivity",
    "local_covariance",
    "shared_commands",
    "distance_bounds",
    "range_bounds",
    "relief_bounds",
    "independent_certificate",
    "global_optimality",
    "statistical_coverage",
]


def _immutable(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _immutable(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(v) for v in value)
    return value


@dataclass(frozen=True, kw_only=True)
class StudyRequest:
    required_evidence: tuple[EvidenceRequirement, ...] = ()
    operation: ClassVar[str]

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _immutable(getattr(self, field.name)))

    def arguments(self):
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "required_evidence"
        }


@dataclass(frozen=True)
class EvaluationRequest(StudyRequest):
    operation: ClassVar[str] = "evaluation"
    realization: Mapping[str, float] | None = None
    backend: Literal["scipy", "cvxpy"] | None = None
    diagnose: bool | None = None
    relief: Mapping | None = None


@dataclass(frozen=True)
class SensitivityRequest(StudyRequest):
    operation: ClassVar[str] = "sensitivity"
    realization: Mapping[str, float] | None = None
    inputs: tuple[str, ...] | None = None
    outputs: tuple[str, ...] | None = None
    operator: Literal["jacobian", "directional"] | None = None
    coordinate_space: Literal["physical", "normalized"] | None = None
    response: Literal["fixed", "optimized"] | None = None
    side: Literal["two_sided", "forward", "backward"] | None = None
    direction: tuple[float, ...] | None = None
    method: str | None = None
    perturbation_scope: str | None = None


@dataclass(frozen=True)
class AuditRequest(StudyRequest):
    operation: ClassVar[str] = "audit"
    backend: Literal["scipy", "cvxpy"] | None = None
    max_vertices: int | None = None
    relief: Mapping | None = None


@dataclass(frozen=True)
class BoundaryRequest(StudyRequest):
    operation: ClassVar[str] = "boundary"
    backend: Literal["scipy", "cvxpy"] | None = None
    distance_tolerance: float | None = None
    constraints: tuple[str, ...] | None = None


@dataclass(frozen=True)
class BreakingRequest(StudyRequest):
    operation: ClassVar[str] = "breaking"
    backend: Literal["scipy", "cvxpy"] | None = None
    distance_tolerance: float | None = None
    constraints: tuple[str, ...] | None = None
    violation_margins: Mapping[str, float] | None = None
    shortfall_margin_mw: float | None = None


@dataclass(frozen=True)
class ControllerSynthesisRequest(StudyRequest):
    operation: ClassVar[str] = "controller_synthesis"
    objective: LinearObjective | ControlTrackingObjective | None = None
    name: str = "Synthesized affine controller"
    max_vertices: int = 256


@dataclass(frozen=True)
class OperatingLimitRequest(StudyRequest):
    operation: ClassVar[str] = "operating_limit"
    parameter: str
    tolerance: float = 1e-6
    max_iterations: int = 64
    max_vertices: int = 256
    backend: Literal["scipy", "cvxpy"] = "scipy"


@dataclass(frozen=True)
class DispatchSensitivityRequest(StudyRequest):
    operation: ClassVar[str] = "dispatch_sensitivity"
    realization: Mapping[str, float]
    changes: tuple[DispatchChange, ...]
    backend: Literal["scipy", "cvxpy"] = "scipy"
    max_refinements: int = 16


@dataclass(frozen=True)
class SharedControlsRequest(StudyRequest):
    operation: ClassVar[str] = "shared_controls"
    domain: UncertaintySet
    operating: OperatingSpec | None = None


@dataclass(frozen=True)
class UncertaintyPropagationRequest(StudyRequest):
    operation: ClassVar[str] = "uncertainty_propagation"
    parameters: tuple  # Ordered IdaesVariable component bindings.
    covariance: ParameterCovariance | tuple[tuple[float, ...], ...]
    operating: OperatingSpec
    realization: Mapping[str, float] | None = None
    outputs: tuple[str, ...] | None = None
    margins: tuple[str, ...] = ()
    steps: Mapping[str, float] | None = None
    relative_tolerance: float = 1e-3
    absolute_tolerance: float = 1e-6


@dataclass(frozen=True)
class StudyCapability(Record):
    operation: str
    status: Literal["supported", "unsupported", "invalid"]
    scope: str
    coverage: str = "none"
    evidence: tuple[EvidenceRequirement, ...] = ()
    dependencies: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def supported(self):
        return self.status == "supported"


class StudyRequestError(ValueError):
    """The explicit request API rejected a study before executing it."""

    def __init__(self, capability):
        self.capability = capability
        super().__init__(capability.scope)


def check_study(target, request):
    """Inspect support without solving, initializing, or traversing scenarios.

    Evidence describes what this method can establish, not a promised outcome.
    Membership, feasible solutions, regularity and actual proof completion remain
    execution checks. Optional dependencies are discovered without importing them.
    """
    from ._study_dispatch import check

    return check(target, request)


def run_study(target, request):
    """Preflight a typed request, then return its existing result family.

    Invalid or unsupported requests raise StudyRequestError with the complete
    capability report. Existing convenience methods retain their result/exception
    conventions. Supported execution can still return an unresolved result.
    """
    from ._study_dispatch import run

    capability = check_study(target, request)
    if not capability.supported:
        raise StudyRequestError(capability)
    return run(target, request)
