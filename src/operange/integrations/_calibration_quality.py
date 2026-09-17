"""Portable local-information and held-out-prediction records."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
import json
from typing import Any, Literal

from ..contract_types import Diagnostic, Record, freeze, nonempty, reference, unique
from ..primitives import finite
from .parmest import CalibrationResult, CalibrationExperimentResult


class _PortableQuality(Record):
    @property
    def ref(self):
        return reference(self.to_dict(), self.schema)

    @property
    def result_id(self):
        return self.ref.artifact_id

    def to_json(self):
        return json.dumps(
            {
                "schema_version": self.schema,
                "result_id": self.result_id,
                **self.to_dict(),
            },
            indent=2,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        if not isinstance(data, dict) or data.pop("schema_version", None) != cls.schema:
            raise ValueError("Unsupported calibration quality schema")
        expected = data.pop("result_id", None)
        try:
            result = cls(**data)
        except (TypeError, KeyError, IndexError) as exc:
            raise ValueError("Invalid calibration quality fields") from exc
        if result.result_id != expected:
            raise ValueError("Calibration quality integrity check failed")
        return result


@dataclass(frozen=True)
class CalibrationSensitivity(Record):
    name: str
    execution: Literal["completed", "unresolved"]
    jacobian: tuple[tuple[float, ...], ...] = ()
    state_condition: float | None = None
    active_limits: tuple[str, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    def _validate(self):
        nonempty(self.name, "experiment name")
        if self.execution == "completed" and not self.jacobian:
            raise ValueError("Completed sensitivity needs measurement rows")
        if self.execution == "unresolved" and (self.jacobian or not self.diagnostics):
            raise ValueError("Unresolved sensitivity needs diagnostics and no Jacobian")
        if self.state_condition is not None and self.state_condition < 1 - 1e-10:
            raise ValueError("State condition number must be at least one")


_LOCAL_SCOPE = {
    "method": "Pyomo automatic differentiation and implicit equality-state sensitivities",
    "scaling": "J[i,j] = (dy[i]/dtheta[j]) * parameter_scale[j] / measurement_sigma[i]",
    "identifiability": "local_numerical_measurement_information_only",
    "structural_identifiability": "not_assessed",
    "confidence_coverage": "not_assessed",
    "accuracy": "not_assessed",
    "state_regularity": "Square state Jacobian; row/column magnitude equilibration; numerical condition limit 1e10.",
    "bounds": "Ambient equality-model derivatives; active bounds and inequalities are reported, not treated as measurements.",
}


@dataclass(frozen=True)
class CalibrationDiagnostics(_PortableQuality):
    calibration: CalibrationResult
    parameter_scales: Mapping[str, float]
    rank_tolerance: float
    weak_threshold: float
    bound_tolerance: float
    execution: Literal["completed", "unresolved"]
    experiments: tuple[CalibrationSensitivity, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    scope: Mapping[str, Any] = field(default_factory=lambda: _LOCAL_SCOPE)
    schema = "calibration_diagnostics/v1"

    def _validate(self):
        if dict(self.scope) != _LOCAL_SCOPE:
            raise ValueError("Local diagnostic scope qualifications must be retained")
        parameters = self.calibration.study["experiments"][0]["parameters"]
        if not parameters or set(self.parameter_scales) != {
            p["component"] for p in parameters
        }:
            raise ValueError("Scales must cover every estimated parameter")
        for value in self.parameter_scales.values():
            if finite(value, "parameter scale") <= 0:
                raise ValueError("Parameter scales must be positive")
        if (
            not 0 < self.rank_tolerance <= self.weak_threshold < 1
            or self.bound_tolerance < 0
        ):
            raise ValueError("Invalid diagnostic thresholds")
        sources = self.calibration.study["experiments"]
        if self.experiments and tuple(e.name for e in self.experiments) != tuple(
            e["name"] for e in sources
        ):
            raise ValueError("Diagnostics must retain every experiment")
        for experiment, source in zip(self.experiments, sources):
            if experiment.execution == "completed" and (
                len(experiment.jacobian) != len(source["observations"])
                or any(len(row) != len(parameters) for row in experiment.jacobian)
            ):
                raise ValueError("Measurement Jacobian has the wrong shape")
        complete = (
            self.calibration.execution == "completed"
            and bool(self.experiments)
            and all(e.execution == "completed" for e in self.experiments)
        )
        if (self.execution == "completed") != complete:
            raise ValueError("Diagnostic execution disagrees with experiment coverage")
        if not complete and not self.diagnostics:
            raise ValueError("Unresolved diagnostics need an explanation")

    @cached_property
    def analysis(self):
        """Reproducible summaries from stored Jacobians, available without Pyomo.

        Directions refer to z[j] = delta(theta[j]) / parameter_scale[j].
        Bases in repeated-singular-value subspaces are not unique.
        """
        import numpy as np

        if self.execution != "completed":
            raise ValueError("Local analysis requires every experiment to be checked")
        names = tuple(p.component for p in self.calibration.estimates)
        matrix = np.asarray([row for e in self.experiments for row in e.jacobian])
        _, values, vectors = np.linalg.svd(
            matrix, full_matrices=matrix.shape[0] < matrix.shape[1]
        )
        values = np.pad(values, (0, len(names) - len(values)))
        largest = values[0]
        rank = int(np.sum(values > largest * self.rank_tolerance))
        directions = []
        for value, vector in zip(values, vectors):
            if vector[np.argmax(np.abs(vector))] < 0:
                vector = -vector
            if value <= largest * self.weak_threshold:
                directions.append(
                    {
                        "singular_value": float(value),
                        "classification": "numerically_null"
                        if value <= largest * self.rank_tolerance
                        else "weak",
                        "coefficients": dict(zip(names, vector.tolist())),
                    }
                )
        return freeze(
            {
                "parameters": names,
                "singular_values": values.tolist(),
                "rank": rank,
                "parameter_count": len(names),
                "condition_number": float(values[0] / values[-1])
                if rank == len(names)
                else None,
                "weak_directions": directions,
            }
        )


@dataclass(frozen=True)
class CalibrationPrediction(Record):
    name: str
    execution: Literal["completed", "unresolved"]
    result: CalibrationExperimentResult | None = None
    solver: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()

    def _validate(self):
        nonempty(self.name, "prediction name")
        if self.result is not None and self.result.name != self.name:
            raise ValueError("Prediction result has a different experiment identity")
        checked = self.result is not None and all(
            c.satisfied for c in self.result.checks
        )
        if self.execution == "completed" and not checked:
            raise ValueError("Completed prediction requires checked source equations")
        if self.execution == "unresolved" and not self.diagnostics:
            raise ValueError("Unresolved prediction needs diagnostics")


_VALIDATION_SCOPE = {
    "prediction": "Fitted parameters and experimental conditions held fixed; measurement labels excluded from the solve objective.",
    "completion": "Checked model predictions, not an accuracy acceptance verdict.",
    "independence": "Caller declares held-out experiments; names must be disjoint from training. Statistical independence is not established.",
    "uniqueness": "Regular local equality-state model required; global uniqueness is not proved.",
    "state_regularity": "Square state Jacobian; row/column magnitude equilibration; numerical condition limit 1e10.",
    "residual_threshold": "Diagnostic threshold in measurement standard deviations, not a confidence test.",
}


@dataclass(frozen=True)
class CalibrationValidation(_PortableQuality):
    calibration: CalibrationResult
    study: Mapping[str, Any]
    residual_threshold: float
    execution: Literal["completed", "unresolved"]
    predictions: tuple[CalibrationPrediction, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
    scope: Mapping[str, Any] = field(default_factory=lambda: _VALIDATION_SCOPE)
    schema = "calibration_validation/v1"

    def _validate(self):
        if dict(self.scope) != _VALIDATION_SCOPE:
            raise ValueError("Validation scope qualifications must be retained")
        if self.calibration.execution != "completed" or self.residual_threshold <= 0:
            raise ValueError(
                "Validation requires a checked fit and a positive threshold"
            )
        sources = self.study["experiments"]
        if not sources or tuple(p.name for p in self.predictions) != tuple(
            s["name"] for s in sources
        ):
            raise ValueError("Validation must retain every held-out experiment")
        unique(tuple(p.name for p in self.predictions), "held-out names")
        if {p.name for p in self.predictions} & {
            e.name for e in self.calibration.experiments
        }:
            raise ValueError("Held-out names overlap the training experiments")
        estimates = {p.component: p for p in self.calibration.estimates}
        for prediction, source in zip(self.predictions, sources):
            if {p["component"]: p["unit"] for p in source["parameters"]} != {
                k: p.unit for k, p in estimates.items()
            }:
                raise ValueError("Held-out parameters differ from calibration")
            result = prediction.result
            if result is None:
                continue
            if tuple(
                {
                    k: getattr(o, k)
                    for k in ("component", "unit", "measured", "standard_deviation")
                }
                for o in result.observations
            ) != tuple(source["observations"]):
                raise ValueError(
                    "Prediction measurements differ from held-out declarations"
                )
            if tuple(
                {k: getattr(c, k) for k in ("component", "side", "unit", "tolerance")}
                for c in result.checks
            ) != tuple(source["checks"]):
                raise ValueError("Prediction source checks are incomplete")
            if set(result.values) != {
                v["component"] for v in source["source"]["variables"]
            }:
                raise ValueError("Prediction variable values are incomplete")
            if any(
                result.values[name] != p.value
                for name, p in estimates.items()
                if name in result.values
            ):
                raise ValueError("Held-out prediction changed fitted parameters")
        complete = all(p.execution == "completed" for p in self.predictions)
        if (self.execution == "completed") != complete:
            raise ValueError("Validation execution disagrees with prediction coverage")
        if not complete and not self.diagnostics:
            raise ValueError("Unresolved validation needs diagnostics")

    @property
    def metrics(self):
        """Residual summaries by measured quantity; never silently drop failures."""
        import math

        if self.execution != "completed":
            raise ValueError("Metrics require all held-out predictions to be checked")
        grouped = {}
        for prediction in self.predictions:
            for observation in prediction.result.observations:
                grouped.setdefault(
                    (observation.component, observation.unit), []
                ).append(observation)
        return tuple(
            freeze(
                {
                    "component": key[0],
                    "unit": key[1],
                    "count": len(rows),
                    "bias": sum(o.residual for o in rows) / len(rows),
                    "rmse": math.sqrt(sum(o.residual**2 for o in rows) / len(rows)),
                    "max_absolute_residual": max(abs(o.residual) for o in rows),
                    "standardized_rmse": math.sqrt(
                        sum(o.standardized_residual**2 for o in rows) / len(rows)
                    )
                    if all(o.standard_deviation is not None for o in rows)
                    else None,
                }
            )
            for key, rows in grouped.items()
        )
