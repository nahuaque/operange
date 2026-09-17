"""Supported calibration through Pyomo parmest, with backend-free result records.

Experiments supply the physical model, fixed conditions and measurement labels.
Operange snapshots them, checks returned numerical candidates, and records the
fit. Importing this module does not load Pyomo, IDAES, pandas or a solver.
"""

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import json
from typing import Any, Literal

from ..contract_types import Diagnostic, Record, freeze, nonempty, reference, unique
from ..primitives import finite

__all__ = [
    "ParmestCalibration",
    "CalibrationResult",
    "CalibrationEstimate",
    "CalibrationObservation",
    "CalibrationCheck",
    "CalibrationExperimentResult",
]


@dataclass(frozen=True)
class CalibrationEstimate(Record):
    component: str
    unit: str
    value: float
    lower: float | None = None
    upper: float | None = None

    def _validate(self):
        nonempty(self.component, "parameter component")
        nonempty(self.unit, "parameter unit")
        if self.lower is not None and self.value < self.lower:
            raise ValueError("estimate is below its declared bound")
        if self.upper is not None and self.value > self.upper:
            raise ValueError("estimate is above its declared bound")


@dataclass(frozen=True)
class CalibrationObservation(Record):
    component: str
    unit: str
    measured: float
    fitted: float
    standard_deviation: float | None = None

    def _validate(self):
        nonempty(self.component, "measurement component")
        nonempty(self.unit, "measurement unit")
        if self.standard_deviation is not None and self.standard_deviation <= 0:
            raise ValueError("measurement standard deviation must be positive")

    @property
    def residual(self):
        """Fitted minus measured, in the declared native physical unit."""
        return self.fitted - self.measured

    @property
    def standardized_residual(self):
        return (
            None
            if self.standard_deviation is None
            else self.residual / self.standard_deviation
        )


@dataclass(frozen=True)
class CalibrationCheck(Record):
    component: str
    side: Literal["eq", "lower", "upper"]
    unit: str
    residual: float
    tolerance: float

    def _validate(self):
        nonempty(self.component, "constraint component")
        nonempty(self.unit, "constraint unit")
        if self.tolerance < 0:
            raise ValueError("constraint tolerance must be nonnegative")

    @property
    def satisfied(self):
        return (
            abs(self.residual) if self.side == "eq" else self.residual
        ) <= self.tolerance


@dataclass(frozen=True)
class CalibrationExperimentResult(Record):
    name: str
    observations: tuple[CalibrationObservation, ...]
    checks: tuple[CalibrationCheck, ...]
    values: Mapping[str, float]

    def _validate(self):
        nonempty(self.name, "experiment name")
        if not self.observations:
            raise ValueError("experiment results need observations")
        unique(tuple(o.component for o in self.observations), "observations")
        unique(tuple((c.component, c.side) for c in self.checks), "physical checks")
        for value in self.values.values():
            finite(value, "returned variable value")


@dataclass(frozen=True)
class CalibrationResult(Record):
    """Numerical calibration evidence, not an identifiability or accuracy proof."""

    study: Mapping[str, Any]
    execution: Literal["completed", "unresolved"]
    estimates: tuple[CalibrationEstimate, ...] = ()
    experiments: tuple[CalibrationExperimentResult, ...] = ()
    objective_value: float | None = None
    solver: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()

    def _validate(self):
        unique(tuple(e.component for e in self.estimates), "parameter estimates")
        unique(tuple(e.name for e in self.experiments), "experiment results")
        if self.execution == "completed":
            if self.objective_value is None or not self.experiments:
                raise ValueError(
                    "completed calibration needs an objective and experiments"
                )
            if any(not c.satisfied for e in self.experiments for c in e.checks):
                raise ValueError(
                    "completed calibration contains violated physical checks"
                )
            self._validate_completed()
        elif self.estimates:
            raise ValueError(
                "unresolved calibration cannot expose applicable estimates"
            )
        if self.objective_value is not None and self.objective_value < 0:
            raise ValueError("least-squares objective cannot be negative")

    def _validate_completed(self):
        """Check portable record consistency, without interpreting source equations."""
        try:
            declared = self.study["experiments"]
            if tuple(e.name for e in self.experiments) != tuple(
                e["name"] for e in declared
            ):
                raise ValueError("experiment results do not match the study")
            expected = {p["component"]: p["unit"] for p in declared[0]["parameters"]}
            if {p.component: p.unit for p in self.estimates} != expected:
                raise ValueError("parameter estimates do not match the study")
            for result, source in zip(self.experiments, declared):
                if tuple(
                    {
                        k: getattr(o, k)
                        for k in ("component", "unit", "measured", "standard_deviation")
                    }
                    for o in result.observations
                ) != tuple(source["observations"]):
                    raise ValueError("observation records do not match the study")
                if tuple(
                    {
                        k: getattr(c, k)
                        for k in ("component", "side", "unit", "tolerance")
                    }
                    for c in result.checks
                ) != tuple(source["checks"]):
                    raise ValueError("physical check records do not match the study")
                if set(result.values) != {
                    v["component"] for v in source["source"]["variables"]
                }:
                    raise ValueError("returned variable records do not match the study")
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("incomplete calibration study manifest") from exc

    @property
    def ref(self):
        return reference(self.to_dict(), "calibration_result/v1")

    @property
    def result_id(self):
        return self.ref.artifact_id

    def to_json(self):
        return json.dumps(
            {
                "schema_version": "calibration_result/v1",
                "result_id": self.result_id,
                **self.to_dict(),
            },
            indent=2,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        if (
            not isinstance(data, dict)
            or data.pop("schema_version", None) != "calibration_result/v1"
        ):
            raise ValueError("Unsupported calibration result schema")
        expected = data.pop("result_id", None)
        try:
            result = cls(**data)
        except TypeError as exc:
            raise ValueError("Invalid calibration result fields") from exc
        if result.result_id != expected:
            raise ValueError("Calibration result integrity check failed")
        return result

    def apply_to(self, adapter, *, parameter_map):
        """Apply explicitly mapped estimates to a fresh IdaesAdapter snapshot.

        Targets must be fixed model parameters, outside scenario inputs and
        actuators. Native units and bounds must match; no conversion is inferred.
        Calibration does not change the domain or operating requirements.
        """
        from ._parmest_backend import apply_calibration

        return apply_calibration(self, adapter, parameter_map)


@dataclass(frozen=True, kw_only=True)
class ParmestCalibration:
    """Snapshot named upstream Experiments for a shared-parameter fit.

    ``experiments`` is an ordered name-to-Experiment mapping. All experiments
    must label the same scalar unknowns. An empty unknown_parameters suffix
    requests reconciliation. Weighted SSE uses native-unit measurement standard
    deviations. Initialization belongs in Experiment.get_labeled_model().
    """

    experiments: InitVar[Mapping[str, object]]
    name: str
    objective: Literal["SSE", "SSE_weighted"] = "SSE_weighted"
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    tolerance: float = 1e-5
    provenance: Mapping[str, Any] = field(default_factory=dict)
    _experiments: tuple = field(init=False, repr=False, compare=False)
    _study: Mapping = field(init=False, repr=False)

    def __post_init__(self, experiments):
        from ._parmest_backend import prepare_study

        nonempty(self.name, "calibration name")
        if self.objective not in ("SSE", "SSE_weighted"):
            raise ValueError("objective must be SSE or SSE_weighted")
        if not 0 <= self.tolerance <= 1e-3:
            raise ValueError("tolerance must be between zero and 1e-3")
        for key in ("solver_options", "provenance"):
            if not isinstance(getattr(self, key), Mapping):
                raise ValueError(f"{key} must be a mapping")
            object.__setattr__(self, key, freeze(getattr(self, key)))
        snapshots, manifest = prepare_study(self, experiments)
        object.__setattr__(self, "_experiments", snapshots)
        object.__setattr__(self, "_study", freeze(manifest))

    @property
    def manifest(self):
        return self._study

    @property
    def ref(self):
        return reference(self._study, "calibration_study/v1")

    def fit(self):
        """Run upstream theta_est and check its returned candidate independently."""
        from ._parmest_backend import fit

        return fit(self)
