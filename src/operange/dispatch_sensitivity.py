"""Portable local value sensitivities of optimized linear dispatch."""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction as F
from functools import cached_property
import json
from typing import Any, ClassVar, Literal

from ._numeric import round_down, round_up
from .contract_types import (
    Diagnostic,
    ProcessContract,
    Record,
    nonempty,
    plain,
    reference,
)
from .controllers import _fields, _read_json
from .engineering_results import EvaluationResult, result_from_json
from .primitives import finite


def _rational(value):
    if (
        not isinstance(value, str)
        or len(value) > 4096
        or any(c not in "-0123456789/" for c in value)
    ):
        raise ValueError("expected a bounded exact rational string")
    try:
        return F(value)
    except ZeroDivisionError as exc:
        raise ValueError("invalid exact rational denominator") from exc


@dataclass(frozen=True)
class DispatchChange(Record):
    """Vary one physical input, operating limit, or control bound.

    Positive change increases the declared numeric value, including for lower
    bounds and >= limits. ``step`` caps the explored distance on each side, in
    ``unit``. All other declarations and operating permissions remain fixed.
    """

    kind: Literal["input", "operating_limit", "control_lower", "control_upper"]
    name: str
    unit: str
    step: float = 1

    def _validate(self):
        nonempty(self.name, "change name")
        nonempty(self.unit, "change unit")
        if self.step <= 0:
            raise ValueError("change step must be positive")

    @property
    def change_id(self):
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class DispatchSlope(Record):
    """A checked affine value segment or a direction with no feasible neighborhood.

    The derivative is with respect to increasing the physical coordinate on
    BOTH sides. The interval is a sufficient validity interval, not necessarily
    the largest one. Proof controls are exact real values, not executable floats.
    """

    side: Literal["backward", "forward"]
    status: Literal["verified", "infeasible", "unresolved"]
    unit: str
    slope_exact: str | None = None
    valid_step: float | None = None
    proof: Mapping[str, Any] = None

    def __post_init__(self):
        if self.proof is None:
            object.__setattr__(self, "proof", {})
        super().__post_init__()

    def _validate(self):
        nonempty(self.unit, "slope unit")
        if self.status == "verified":
            finite(float(_rational(self.slope_exact)), "slope")
            if self.valid_step is None or self.valid_step <= 0 or not self.proof:
                raise ValueError("verified slope requires an interval and proof")
        elif self.slope_exact is not None:
            raise ValueError("unverified direction cannot carry a slope")
        if self.status == "infeasible" and (
            not self.proof or self.valid_step is None or self.valid_step <= 0
        ):
            raise ValueError(
                "infeasible direction requires a checked interval and proof"
            )
        if self.status == "unresolved" and (self.proof or self.valid_step is not None):
            raise ValueError("unresolved direction cannot assert a validity interval")

    @property
    def value(self):
        return None if self.slope_exact is None else float(_rational(self.slope_exact))

    @property
    def lower(self):
        return (
            None
            if self.slope_exact is None
            else round_down(_rational(self.slope_exact))
        )

    @property
    def upper(self):
        return (
            None if self.slope_exact is None else round_up(_rational(self.slope_exact))
        )

    @property
    def base_value(self):
        """Rounded mathematical optimum from the local proof, when available."""
        return (
            None
            if self.status != "verified"
            else float(_rational(self.proof["base_value_exact"]))
        )

    @property
    def supporting_constraints(self):
        return tuple(
            ref
            for ref, value in zip(
                self.proof.get("row_refs", ()), self.proof.get("multipliers_exact", ())
            )
            if _rational(value) > 0
        )


@dataclass(frozen=True)
class DispatchSensitivity(Record):
    change: DispatchChange
    backward: DispatchSlope
    forward: DispatchSlope

    def _validate(self):
        if self.backward.side != "backward" or self.forward.side != "forward":
            raise ValueError("dispatch slope sides are reversed")
        if self.backward.unit != self.forward.unit:
            raise ValueError("dispatch slope units differ")
        if any(
            s.valid_step is not None and s.valid_step > self.change.step
            for s in (self.backward, self.forward)
        ):
            raise ValueError("validity interval exceeds the requested step")

    @property
    def resolution(self):
        if "unresolved" in (self.backward.status, self.forward.status):
            return "unresolved"
        if self.backward.status == self.forward.status == "verified":
            return (
                "differentiable"
                if _rational(self.backward.slope_exact)
                == _rational(self.forward.slope_exact)
                else "kink"
            )
        return (
            "one_sided"
            if "verified" in (self.backward.status, self.forward.status)
            else "isolated_feasible_point"
        )


@dataclass(frozen=True)
class DispatchSensitivityResult(Record):
    """Objective-value slopes with exact primal/dual certificates on local rays.

    These are ambient derivatives of the real-valued optimization problem, not
    derivatives of a selected command vector or guarantees over uncertainty.
    Use verify_result() to independently check the exported mathematics.
    """

    contract: ProcessContract
    request: Mapping[str, Any]
    execution: Literal["completed", "unresolved", "unsupported", "invalid"]
    sensitivities: tuple[DispatchSensitivity, ...] = ()
    base: EvaluationResult | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    solver_attempts: tuple[Mapping[str, Any], ...] = ()
    schema_version: ClassVar[str] = "dispatch_sensitivity/v1"

    def _validate(self):
        if self.base is not None and self.base.contract != self.contract:
            raise ValueError("dispatch sensitivity base has another contract")
        if self.sensitivities and (
            self.base is None or self.base.payload.feasibility != "feasible"
        ):
            raise ValueError("dispatch slopes require a feasible base evaluation")
        if self.execution == "completed" and (
            not self.sensitivities
            or any(s.resolution == "unresolved" for s in self.sensitivities)
        ):
            raise ValueError("completed sensitivity requires checked directions")
        if self.execution in ("unsupported", "invalid") and self.sensitivities:
            raise ValueError("rejected queries cannot assert sensitivities")

    @property
    def contract_ref(self):
        return self.contract.ref

    def _body(self, compact=False):
        return {
            "schema_version": self.schema_version,
            "contract": self.contract.to_dict(),
            "request": plain(self.request),
            "execution": self.execution,
            "sensitivities": [s.to_dict() for s in self.sensitivities],
            "base": None if self.base is None else self.base.to_dict(compact=compact),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "solver_attempts": plain(self.solver_attempts),
        }

    @cached_property
    def ref(self):
        return reference(self._body(), self.schema_version)

    def to_dict(self, *, compact=False):
        return {**self._body(compact), "result_id": self.ref.artifact_id}

    def to_json(self, *, compact=False, indent=2):
        return json.dumps(
            self.to_dict(compact=compact),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, document):
        data = _read_json(document)
        _fields(
            data,
            (
                "schema_version",
                "contract",
                "request",
                "execution",
                "sensitivities",
                "base",
                "diagnostics",
                "solver_attempts",
                "result_id",
            ),
        )
        if data["schema_version"] != cls.schema_version:
            raise ValueError("unsupported dispatch sensitivity schema")
        result = cls(
            data["contract"],
            data["request"],
            data["execution"],
            data["sensitivities"],
            None
            if data["base"] is None
            else result_from_json(json.dumps(data["base"])),
            data["diagnostics"],
            data["solver_attempts"],
        )
        if result.ref.artifact_id != data["result_id"]:
            raise ValueError("dispatch sensitivity identity mismatch")
        return result


def dispatch_sensitivity(
    claim, realization, *, changes, backend="scipy", max_refinements=16
):
    """Check one-coordinate local objective slopes with bounded numerical search."""
    from ._dispatch_sensitivity import analyze

    return analyze(claim, realization, changes, backend, max_refinements)
