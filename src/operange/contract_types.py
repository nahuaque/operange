"""Immutable engineering values and evidence, independent of presentation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
import hashlib
import json
from types import MappingProxyType, UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .primitives import finite


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if is_dataclass(value):
        return {item.name: plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    return value


def digest(value) -> str:
    encoded = json.dumps(
        plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("record keys must be strings")
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        return finite(value, "record value")
    raise ValueError(f"unsupported record value: {type(value).__name__}")


def snapshot(value):
    """Retain invalid request inputs without exporting non-finite JSON numbers."""
    try:
        return freeze(value)
    except ValueError:
        if isinstance(value, Mapping):
            return freeze({str(k): snapshot(v) for k, v in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(snapshot(v) for v in value)
        return freeze({"invalid_type": type(value).__name__, "value": str(value)})


def _coerce(value, annotation):
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation in (Any, object):
        return freeze(value)
    if origin in (Union, UnionType):
        if type(None) in args:
            if value is None:
                return None
            remaining = tuple(choice for choice in args if choice is not type(None))
            if len(remaining) == 1:
                return _coerce(value, remaining[0])
        for choice in args:
            try:
                return _coerce(value, choice)
            except (ValueError, TypeError):
                pass
        raise ValueError("value does not match its declared type")
    if annotation is type(None):
        if value is not None:
            raise ValueError("expected null")
        return None
    if origin is Literal:
        if not any(type(value) is type(option) and value == option for option in args):
            raise ValueError(f"expected one of {args}")
        return value
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            raise ValueError("expected an array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(item, args[0]) for item in value)
        if len(value) != len(args):
            raise ValueError("array has the wrong length")
        return tuple(_coerce(item, typ) for item, typ in zip(value, args))
    if origin is Mapping:
        if not isinstance(value, Mapping):
            raise ValueError("expected a mapping")
        return freeze(value)
    if annotation is float:
        return finite(value, "numeric field")
    if annotation in (str, int, bool):
        if type(value) is not annotation:
            raise ValueError(f"expected {annotation.__name__}")
        return value
    if isinstance(annotation, type) and is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if isinstance(value, Mapping):
            try:
                return annotation(**value)
            except TypeError as exc:
                raise ValueError(f"invalid {annotation.__name__} fields") from exc
    raise ValueError(f"expected {annotation}")


@lru_cache(maxsize=None)
def _hints(cls):
    return get_type_hints(cls)


class Record:
    def __post_init__(self):
        annotations = _hints(type(self))
        for item in fields(self):
            try:
                value = _coerce(getattr(self, item.name), annotations[item.name])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{type(self).__name__}.{item.name}: {exc}") from exc
            object.__setattr__(self, item.name, value)
        self._validate()

    def _validate(self):
        pass

    def to_dict(self):
        return plain(self)


def nonempty(value, name):
    if not value or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def unique(values, name):
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {name}")


def numeric_shape(value):
    if isinstance(value, tuple):
        shapes = [numeric_shape(item) for item in value]
        if shapes and any(shape != shapes[0] for shape in shapes):
            raise ValueError("ragged numeric array")
        return (len(value), *(shapes[0] if shapes else ()))
    finite(value, "quantity value")
    return ()


@dataclass(frozen=True)
class ArtifactRef(Record):
    artifact_id: str
    schema_version: str
    digest: str

    def _validate(self):
        if len(self.digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.digest
        ):
            raise ValueError("invalid SHA-256 digest")
        if self.artifact_id != f"sha256:{self.digest}":
            raise ValueError("artifact identity does not match digest")
        nonempty(self.schema_version, "schema_version")


def reference(value, schema_version):
    fingerprint = digest(value)
    return ArtifactRef(f"sha256:{fingerprint}", schema_version, fingerprint)


@dataclass(frozen=True)
class Axis(Record):
    axis_id: str
    coordinates: tuple[str, ...]

    def _validate(self):
        nonempty(self.axis_id, "axis_id")
        if not self.coordinates:
            raise ValueError("axis must have coordinates")
        unique(self.coordinates, "axis coordinates")
        for coordinate in self.coordinates:
            nonempty(coordinate, "coordinate")


@dataclass(frozen=True)
class QuantitySpec(Record):
    quantity_id: str
    unit: str
    physical_kind: str
    role: Literal["input", "design", "state", "control", "output", "residual"]
    axes: tuple[Axis, ...] = ()

    def _validate(self):
        for name in ("quantity_id", "unit", "physical_kind"):
            nonempty(getattr(self, name), name)
        unique(tuple(a.axis_id for a in self.axes), "quantity axes")


@dataclass(frozen=True)
class QuantityValue(Record):
    quantity_ref: str
    value: object

    def _validate(self):
        nonempty(self.quantity_ref, "quantity_ref")
        numeric_shape(self.value)


@dataclass(frozen=True)
class ConstraintSpec(Record):
    constraint_id: str
    residual_ref: str
    kind: Literal["equation", "operating_limit", "requirement"]
    relation: Literal["eq", "le"]
    expression: str
    tolerance: float

    def _validate(self):
        for name in ("constraint_id", "residual_ref", "expression"):
            nonempty(getattr(self, name), name)
        if self.tolerance < 0:
            raise ValueError("tolerance must be nonnegative")


@dataclass(frozen=True)
class ProcessContract(Record):
    model_id: str
    model: Mapping[str, Any]
    design: Mapping[str, Any]
    domain: Mapping[str, Any]
    operating: Mapping[str, Any]
    quantities: tuple[QuantitySpec, ...]
    constraints: tuple[ConstraintSpec, ...]
    numerical_policy: Mapping[str, Any]

    def _validate(self):
        nonempty(self.model_id, "model_id")
        if not self.quantities:
            raise ValueError("quantity registry must not be empty")
        unique(tuple(q.quantity_id for q in self.quantities), "quantity identities")
        unique(
            tuple(c.constraint_id for c in self.constraints), "constraint identities"
        )
        axes = {}
        for quantity in self.quantities:
            for axis in quantity.axes:
                if axis.axis_id in axes and axes[axis.axis_id] != axis:
                    raise ValueError("conflicting axis definitions")
                axes[axis.axis_id] = axis
        for constraint in self.constraints:
            if self.quantity(constraint.residual_ref).role != "residual":
                raise ValueError("constraint must refer to a residual quantity")

    @property
    def ref(self):
        return reference(self, "process_contract/v1")

    def quantity(self, identity):
        for quantity in self.quantities:
            if quantity.quantity_id == identity:
                return quantity
        raise ValueError(f"unknown quantity: {identity}")

    def check_value(self, value: QuantityValue):
        quantity = self.quantity(value.quantity_ref)
        if numeric_shape(value.value) != tuple(
            len(a.coordinates) for a in quantity.axes
        ):
            raise ValueError(f"wrong shape for {value.quantity_ref}")


@dataclass(frozen=True)
class Measurement(Record):
    name: str
    value: float
    unit: str

    def _validate(self):
        nonempty(self.name, "measurement name")
        nonempty(self.unit, "measurement unit")


@dataclass(frozen=True)
class Evidence(Record):
    evidence_id: str
    subject: str
    method: str
    outcome: Literal["verified", "unresolved", "failed"]
    measurements: tuple[Measurement, ...] = ()
    tolerances: tuple[Measurement, ...] = ()
    assumptions: tuple[str, ...] = ()
    backend: str | None = None
    details: Mapping[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            object.__setattr__(self, "details", {})
        super().__post_init__()

    def _validate(self):
        for name in ("evidence_id", "subject", "method"):
            nonempty(getattr(self, name), name)
        if any(m.value < 0 for m in self.tolerances):
            raise ValueError("negative evidence tolerance")


@dataclass(frozen=True)
class Diagnostic(Record):
    code: str
    subject: str
    message: str

    def _validate(self):
        nonempty(self.code, "diagnostic code")
        nonempty(self.subject, "diagnostic subject")


@dataclass(frozen=True)
class Bound(Record):
    state: Literal["finite", "unbounded", "unknown"]
    unit: str
    value: float | None = None
    evidence_refs: tuple[str, ...] = ()

    def _validate(self):
        nonempty(self.unit, "bound unit")
        if (self.state == "finite") != (self.value is not None):
            raise ValueError("only finite bounds have a value")
        if self.state != "unknown" and not self.evidence_refs:
            raise ValueError("a bound needs supporting evidence")


@dataclass(frozen=True)
class Membership(Record):
    status: Literal["inside", "outside", "unknown", "not_checked"]
    domain_ref: str = "domain"
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintCheck(Record):
    constraint_ref: str
    assessment: Literal["satisfied", "violated", "unknown", "not_evaluated"]
    residual: QuantityValue | None = None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Objective(Record):
    quantity_ref: str
    sense: Literal["minimize", "maximize"]
    attained_value: float | None
    lower: Bound
    upper: Bound
    attainment: Literal["attained", "not_attained", "unknown"]
    optimality: Literal["verified", "local", "unknown"]

    def _validate(self):
        if self.lower.unit != self.upper.unit:
            raise ValueError("objective bounds have different units")
        if self.lower.value is not None and self.upper.value is not None:
            if self.lower.value > self.upper.value:
                raise ValueError("objective bounds are reversed")
        if self.attainment == "attained" and self.attained_value is None:
            raise ValueError("attained objective needs a value")


@dataclass(frozen=True)
class EvaluationPayload(Record):
    feasibility: Literal["feasible", "infeasible", "unknown", "not_evaluated"]
    membership: Membership
    values: tuple[QuantityValue, ...] = ()
    constraint_checks: tuple[ConstraintCheck, ...] = ()
    objective: Objective | None = None
    included_constraints: tuple[str, ...] = ()

    def _validate(self):
        unique(tuple(v.quantity_ref for v in self.values), "quantity values")
        unique(
            tuple(c.constraint_ref for c in self.constraint_checks), "constraint checks"
        )
        unique(self.included_constraints, "included constraints")


@dataclass(frozen=True)
class Derivative(Record):
    operator: Literal["jacobian", "directional"]
    output_coordinates: tuple[str, ...]
    input_coordinates: tuple[str, ...]
    coordinate_space: Literal["physical", "normalized"]
    values: object
    units: tuple[tuple[str, ...], ...]
    input_units: tuple[str, ...]
    output_units: tuple[str, ...]
    origins: tuple[float, ...]
    scales: tuple[float, ...]
    side: Literal["two_sided", "forward", "backward"]
    direction: tuple[float, ...] | None = None

    def _validate(self):
        n, m = len(self.output_coordinates), len(self.input_coordinates)
        if not n or not m:
            raise ValueError("derivative coordinates must be nonempty")
        unique(self.output_coordinates, "derivative outputs")
        unique(self.input_coordinates, "derivative inputs")
        for values in (self.origins, self.scales, self.input_units):
            if len(values) != m:
                raise ValueError("input coordinate metadata has wrong length")
        if len(self.output_units) != n or any(s <= 0 for s in self.scales):
            raise ValueError("invalid derivative units/scales")
        shape = (n, m) if self.operator == "jacobian" else (n,)
        if numeric_shape(self.values) != shape:
            raise ValueError("derivative matrix has wrong shape")
        if self.operator == "directional":
            if self.direction is None or len(self.direction) != m:
                raise ValueError("directional derivative needs a complete direction")
            expected_units = tuple((unit,) for unit in self.output_units)
        else:
            if self.direction is not None:
                raise ValueError("Jacobian must not carry a direction")
            expected_units = tuple(
                tuple(f"{out}/{inp}" for inp in self.input_units)
                for out in self.output_units
            )
        if self.units != expected_units:
            raise ValueError("derivative units do not match coordinates")


@dataclass(frozen=True)
class SensitivityPayload(Record):
    availability: Literal["available", "undefined", "unknown", "not_evaluated"]
    base_evaluation_ref: ArtifactRef | None = None
    response_contract: Mapping[str, Any] = None
    derivative: Derivative | None = None
    method: Mapping[str, Any] = None
    validity: Mapping[str, Any] = None

    def __post_init__(self):
        for name in ("response_contract", "method", "validity"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, {})
        super().__post_init__()

    def _validate(self):
        if (self.availability == "available") != (self.derivative is not None):
            raise ValueError("only an available sensitivity may carry a derivative")
        if self.availability == "available" and self.base_evaluation_ref is None:
            raise ValueError("derivative needs its base evaluation")


@dataclass(frozen=True)
class Coverage(Record):
    method: Literal["analytical_domain", "complete_finite", "partial", "none"]
    domain_ref: str = "domain"
    evaluated_support: Mapping[str, Any] = None
    unexplored_support: Mapping[str, Any] = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("evaluated_support", "unexplored_support"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, {})
        super().__post_init__()


@dataclass(frozen=True)
class Cardinality(Record):
    lower: Bound
    upper: Bound
    criterion: Literal["scenario_count"] = "scenario_count"

    def _validate(self):
        for bound in (self.lower, self.upper):
            if bound.unit != "count" or bound.value is None or bound.value < 1:
                raise ValueError("cardinality requires positive count bounds")
            if int(bound.value) != bound.value:
                raise ValueError("cardinality must be an integer")
        if self.lower.value > self.upper.value:
            raise ValueError("cardinality bounds are reversed")


@dataclass(frozen=True)
class Witness(Record):
    kind: Literal[
        "realization",
        "path",
        "incompatible_futures",
        "individual_infeasibility",
        "fixed_policy_failure",
    ]
    realizations: tuple[Mapping[str, Any], ...]
    affected_constraints: tuple[str, ...]
    membership_evidence_refs: tuple[str, ...]
    recourse_evidence_refs: tuple[str, ...]
    shared_information_node: str | None = None
    details: Mapping[str, Any] = None
    cardinality: Cardinality | None = None

    def __post_init__(self):
        if self.details is None:
            object.__setattr__(self, "details", {})
        super().__post_init__()

    def _validate(self):
        if (
            not self.realizations
            or not self.membership_evidence_refs
            or not self.recourse_evidence_refs
        ):
            raise ValueError("witness requires realizations and both proof obligations")
        if self.kind == "incompatible_futures" and (
            len(self.realizations) < 2 or self.shared_information_node is None
        ):
            raise ValueError(
                "causal conflict needs futures and a shared information node"
            )
        if self.cardinality and self.cardinality.upper.value != len(self.realizations):
            raise ValueError("cardinality does not match witness size")


@dataclass(frozen=True)
class Search(Record):
    target_kind: Literal["breaking_shortfall", "nonpositive_margin"]
    relation: Literal["ge"]
    threshold: QuantityValue
    severity: Mapping[str, Any]
    resolution: Literal["minimum_verified", "bounded", "unreachable", "unresolved"]
    lower: Bound
    upper: Bound
    tolerance: float
    candidate_evaluation_ref: ArtifactRef | None = None
    candidate_distance: float | None = None
    attainment: Literal["attained", "not_attained", "unknown"] = "unknown"
    evidence_refs: tuple[str, ...] = ()

    def _validate(self):
        if self.tolerance < 0 or self.lower.unit != self.upper.unit:
            raise ValueError("invalid search tolerance or units")
        if self.lower.value is not None and self.upper.value is not None:
            if self.lower.value > self.upper.value:
                raise ValueError("search bounds are reversed")
        if self.candidate_distance is not None:
            if self.candidate_distance < 0 or self.candidate_evaluation_ref is None:
                raise ValueError("candidate distance requires an evaluated candidate")
        if self.resolution == "minimum_verified":
            if self.attainment != "attained" or self.candidate_distance is None:
                raise ValueError("verified minimum must be attained")
            if self.lower.value is None or self.upper.value is None:
                raise ValueError("verified minimum requires both bounds")
            if self.upper.value - self.lower.value > self.tolerance:
                raise ValueError("minimum bounds do not close")
            if abs(self.upper.value - self.candidate_distance) > self.tolerance:
                raise ValueError("candidate and minimum disagree")
        if self.resolution == "unreachable" and self.candidate_distance is not None:
            raise ValueError("unreachable target cannot have a candidate")


@dataclass(frozen=True)
class RobustnessPayload(Record):
    verdict: Literal["pass", "fail", "inconclusive", "not_assessed"]
    coverage: Coverage
    evaluation_refs: tuple[ArtifactRef, ...] = ()
    witness: Witness | None = None
    search: Search | None = None

    def _validate(self):
        if self.verdict == "fail" and self.witness is None:
            raise ValueError("failure needs a validated witness")
        if self.witness is not None and self.verdict != "fail":
            raise ValueError("validated witness must refute the claim")
        if self.verdict == "pass" and self.coverage.method not in (
            "analytical_domain",
            "complete_finite",
        ):
            raise ValueError("pass requires full-domain coverage")
        if self.verdict == "pass" and not self.coverage.evidence_refs:
            raise ValueError("pass requires coverage evidence")
