"""Supported calibration through Pyomo parmest, with backend-free result records.

Experiments supply the physical model, fixed conditions and measurement labels.
Operange snapshots them, checks returned numerical candidates, and records the
fit. Importing this module does not load Pyomo, IDAES, pandas or a solver.
"""

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import json
from typing import Any, Literal, TYPE_CHECKING

from ..contract_types import Diagnostic, Record, freeze, nonempty, reference, unique
from ..primitives import finite

__all__ = [
    "ParmestCalibration",
    "CalibrationResult",
    "CalibrationEstimate",
    "CalibrationObservation",
    "CalibrationCheck",
    "CalibrationExperimentResult",
    "CalibrationDraw",
    "CalibrationBootstrap",
    "CalibrationDiagnostics",
    "CalibrationSensitivity",
    "CalibrationValidation",
    "CalibrationPrediction",
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

    def bootstrap(self, *, samples, seed):
        """Joint upstream bootstrap draws with fixed-parameter physical replays.

        Resample whole experiments with replacement. An explicit seed is
        required. The result is empirical numerical evidence, not a confidence
        region or an identifiability assessment.
        """
        from ._parmest_bootstrap import bootstrap

        return bootstrap(self, samples=samples, seed=seed)

    def diagnose(
        self,
        fitted=None,
        *,
        parameter_scales,
        rank_tolerance=1e-8,
        weak_threshold=1e-3,
        bound_tolerance=1e-6,
    ):
        """Assess local measurement information in explicit parameter scales.

        Uses Pyomo derivatives of the source equations at the checked fit.
        Measurement standard deviations are required for every observation.
        Full local rank is not a structural identifiability or accuracy proof.
        """
        from ._calibration_diagnostics import diagnose

        return diagnose(
            self,
            fitted,
            parameter_scales=parameter_scales,
            rank_tolerance=rank_tolerance,
            weak_threshold=weak_threshold,
            bound_tolerance=bound_tolerance,
        )

    def validate(self, fitted, *, experiments, residual_threshold=3.0):
        """Predict explicitly held-out experiments with fitted parameters fixed.

        Observation labels never enter the prediction objective. The fixed-
        parameter model must determine its states through regular equations.
        Completion means predictions were checked, not that the fit is accurate.
        """
        from ._calibration_diagnostics import validate

        return validate(
            self, fitted, experiments=experiments, residual_threshold=residual_threshold
        )


@dataclass(frozen=True)
class CalibrationDraw(Record):
    """One complete parameter vector and its resampled experiment replay."""

    name: str
    experiment_names: tuple[str, ...]
    estimates: tuple[CalibrationEstimate, ...] = ()
    replay: CalibrationResult | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    def _validate(self):
        nonempty(self.name, "draw name")
        unique(tuple(p.component for p in self.estimates), "draw parameters")
        for name in self.experiment_names:
            nonempty(name, "resampled experiment name")
        if self.checked and (not self.estimates or not self.experiment_names):
            raise ValueError("checked draws need estimates and resampled experiments")

    @property
    def checked(self):
        return self.replay is not None and self.replay.execution == "completed"


_BOOTSTRAP_SCOPE = {
    "coverage": "finite_joint_parameter_draws",
    "confidence_level": None,
    "identifiability": "not_assessed",
    "sampling": "Whole experiments with replacement, conditioned by parmest on more distinct experiments than estimated parameters.",
    "assumptions": "Caller treats experiments as exchangeable sampling units; within-experiment measurements stay together.",
    "replay": "Resampled models re-solved at each fixed parameter vector and source residuals checked; the original bootstrap state and objective are not exposed by the upstream API.",
}


@dataclass(frozen=True)
class CalibrationBootstrap(Record):
    """Portable joint empirical draws, retaining failures and multiplicity."""

    calibration: CalibrationResult
    requested_samples: int
    seed: int
    execution: Literal["completed", "unresolved"]
    draws: tuple[CalibrationDraw, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    method: Literal["parmest_experiment_bootstrap"] = "parmest_experiment_bootstrap"
    scope: Mapping[str, Any] = field(default_factory=lambda: _BOOTSTRAP_SCOPE)

    def _validate(self):
        if dict(self.scope) != _BOOTSTRAP_SCOPE:
            raise ValueError(
                "Bootstrap scope must retain empirical finite-sample qualifications"
            )
        if self.requested_samples < 2 or not 0 <= self.seed < 2**32:
            raise ValueError(
                "bootstrap needs at least two samples and a seed in [0, 2**32)"
            )
        unique(tuple(d.name for d in self.draws), "bootstrap draws")
        if len(self.draws) not in (0, self.requested_samples):
            raise ValueError(
                "bootstrap must retain every returned draw, including failures"
            )
        expected = {p.component: p.unit for p in self.calibration.estimates}
        originals = self.calibration.study["experiments"]
        names = {e["name"] for e in originals}
        for draw in self.draws:
            if (
                draw.estimates
                and {p.component: p.unit for p in draw.estimates} != expected
            ):
                raise ValueError(
                    "bootstrap parameter coordinates differ from calibration"
                )
            if draw.experiment_names and (
                len(draw.experiment_names) != len(originals)
                or not set(draw.experiment_names) <= names
            ):
                raise ValueError(
                    "bootstrap experiment identities differ from calibration"
                )
            if draw.checked:
                if draw.replay.estimates or len(draw.replay.experiments) != len(
                    originals
                ):
                    raise ValueError(
                        "bootstrap replay must fix parameters and retain every resampled experiment"
                    )
                if (
                    draw.replay.study["provenance"].get("bootstrap_calibration")
                    != self.calibration.ref.to_dict()
                ):
                    raise ValueError("bootstrap replay refers to another calibration")
                original_by_name = {e["name"]: e for e in originals}
                for position, (name, replay_source) in enumerate(
                    zip(draw.experiment_names, draw.replay.study["experiments"])
                ):
                    original = original_by_name[name]
                    if (
                        replay_source["name"] != f"{position}:{name}"
                        or replay_source["observations"] != original["observations"]
                        or replay_source["source"]["constraints"]
                        != original["source"]["constraints"]
                    ):
                        raise ValueError(
                            "bootstrap replay differs from its resampled experiment"
                        )
                    variables = {
                        v["component"]: v for v in replay_source["source"]["variables"]
                    }
                    parameters = {
                        p["component"]: p for p in replay_source["source"]["parameters"]
                    }
                    for p in draw.estimates:
                        if p.component in variables:
                            declared = variables[p.component]
                            valid = declared["fixed"] and declared["value"] == p.value
                        else:
                            declared = parameters.get(p.component)
                            valid = declared is not None and tuple(
                                declared["values"]
                            ) == ({"index": None, "value": p.value},)
                        if not valid:
                            raise ValueError(
                                "bootstrap replay did not fix the declared parameter vector"
                            )
                for experiment in draw.replay.experiments:
                    for p in draw.estimates:
                        if (
                            p.component in experiment.values
                            and experiment.values[p.component] != p.value
                        ):
                            raise ValueError(
                                "bootstrap replay changed a fitted parameter"
                            )
        complete = (
            self.calibration.execution == "completed"
            and bool(expected)
            and len(self.draws) == self.requested_samples
            and all(d.checked for d in self.draws)
        )
        if (self.execution == "completed") != complete:
            raise ValueError(
                "bootstrap execution disagrees with its coverage and replay checks"
            )

    @property
    def ref(self):
        return reference(self.to_dict(), "calibration_bootstrap/v1")

    @property
    def result_id(self):
        return self.ref.artifact_id

    def _require_complete(self):
        if self.execution != "completed":
            raise ValueError(
                "Joint propagation requires every requested bootstrap draw to be checked; inspect unresolved draws"
            )

    def as_domain(self):
        """A FiniteSet of whole vectors; duplicate draws retain their labels."""
        from ..domains import Coordinate, ParameterSpace, FiniteSet, Scenario

        self._require_complete()
        result_id = self.result_id
        space = ParameterSpace(
            tuple(
                Coordinate(
                    p.component, p.unit, "Joint bootstrap parameter", nominal=p.value
                )
                for p in self.calibration.estimates
            )
        )
        return FiniteSet(
            space,
            tuple(
                Scenario(
                    d.name,
                    {p.component: p.value for p in d.estimates},
                    f"Empirical joint draw from {result_id}; no confidence coverage asserted",
                )
                for d in self.draws
            ),
        )

    @property
    def statistics(self):
        """Empirical mean and full sample covariance in declared parameter order."""
        import numpy as np

        self._require_complete()
        names = tuple(p.component for p in self.calibration.estimates)
        rows = np.asarray(
            [
                [
                    next(p.value for p in d.estimates if p.component == name)
                    for name in names
                ]
                for d in self.draws
            ]
        )
        mean = rows.mean(axis=0)
        centered = rows - mean
        covariance = centered.T @ centered / (len(rows) - 1)
        return freeze(
            {
                "parameters": names,
                "units": tuple(p.unit for p in self.calibration.estimates),
                "mean": mean.tolist(),
                "covariance": covariance.tolist(),
                "samples": len(rows),
                "interpretation": "Empirical sample moments only; no Gaussian model or confidence region inferred.",
            }
        )

    def as_claim(
        self,
        adapter,
        operating_cases,
        *,
        parameter_map,
        fixed_controls=None,
        fully_observed=False,
    ):
        """Cross whole draws with finite cases on a fresh IDAES model snapshot.

        Supply every fixed actuator command, or explicitly request ideal
        dispatch observing every input including the actual parameter vector.
        """
        from ._calibration_propagation import as_claim

        return as_claim(
            self,
            adapter,
            operating_cases,
            parameter_map=parameter_map,
            fixed_controls=fixed_controls,
            fully_observed=fully_observed,
        )

    def optimize_shared_controls(self, adapter, operating_cases, *, parameter_map):
        """Choose common commands using the worst-case source objective.

        All checked joint draws are crossed with all operating cases. Commands
        observe neither the operating case nor the latent parameter vector.
        The result retains numerical source checks, not an optimality proof.
        """
        from ._calibration_propagation import prepare_joint

        propagated, domain = prepare_joint(
            self,
            adapter,
            operating_cases,
            parameter_map=parameter_map,
            dispatch="optimized_shared_actuator_commands",
        )
        return propagated.optimize_shared_controls(domain)

    def to_json(self):
        return json.dumps(
            {
                "schema_version": "calibration_bootstrap/v1",
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
            or data.pop("schema_version", None) != "calibration_bootstrap/v1"
        ):
            raise ValueError("Unsupported calibration bootstrap schema")
        expected = data.pop("result_id", None)
        try:
            result = cls(**data)
        except (TypeError, KeyError) as exc:
            raise ValueError("Invalid calibration bootstrap fields") from exc
        if result.result_id != expected:
            raise ValueError("Calibration bootstrap integrity check failed")
        return result


# The quality records use CalibrationResult. Lazy re-exports keep either module
# independently importable and preserve optional-backend-free public imports.
if TYPE_CHECKING:
    from ._calibration_quality import (
        CalibrationDiagnostics,
        CalibrationSensitivity,
        CalibrationValidation,
        CalibrationPrediction,
    )


def __getattr__(name):
    if name in {
        "CalibrationDiagnostics",
        "CalibrationSensitivity",
        "CalibrationValidation",
        "CalibrationPrediction",
    }:
        from . import _calibration_quality

        return getattr(_calibration_quality, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
