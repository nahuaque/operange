"""Portable local covariance diagnostics; importing these needs no simulator."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
import json
from typing import Any, Literal

import numpy as np

from ..contract_types import Diagnostic, Record, freeze, nonempty, reference, unique
from ..engineering_results import EvaluationResult
from ..primitives import finite
from ..parameters import covariance_matrix


@dataclass(frozen=True)
class IdaesUncertainParameter(Record):
    name: str
    component: str
    unit: str
    value: float

    def _validate(self):
        for key in ("name", "component", "unit"):
            nonempty(getattr(self, key), key)


@dataclass(frozen=True)
class IdaesResponse(Record):
    name: str
    component: str
    unit: str
    kind: Literal["output", "control", "margin"]
    side: Literal["lower", "upper"] | None = None

    def _validate(self):
        for key in ("name", "component", "unit"):
            nonempty(getattr(self, key), key)
        if (self.kind == "margin") != (self.side is not None):
            raise ValueError("Only margins must declare a bound side")


@dataclass(frozen=True)
class IdaesPerturbation(Record):
    parameter: str
    offset: float
    evaluation: EvaluationResult | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    def _validate(self):
        nonempty(self.parameter, "perturbed parameter")
        if not self.offset:
            raise ValueError("A perturbation needs a nonzero offset")
        if not self.checked and not self.diagnostics:
            raise ValueError("Unresolved perturbations require diagnostics")

    @property
    def checked(self):
        return (
            not self.diagnostics
            and self.evaluation is not None
            and self.evaluation.payload.feasibility == "feasible"
        )


_SCOPE = {
    "coverage": "First-order local response at one checked operating point; no uncertainty-region audit.",
    "covariance": "Full supplied parameter covariance; no Gaussian distribution or confidence coverage inferred.",
    "fixed_controls": "Ambient equality-state response with every actuator command fixed; inequalities are reported, not imposed on derivatives.",
    "reoptimized_controls": "Ideal reoptimization observing the actual parameter vector at each perturbation; not a deployable controller for latent parameters.",
    "validity": "Nonlinear curvature, branch changes and changing active constraints can invalidate the local approximation; successful numerical checks do not prove differentiability.",
    "margins": "Signed native-unit slack, positive inside a limit; not a distance in parameter space or a failure probability.",
    "optimality": "not_certified",
    "certificate": "not_independently_verified",
}


@dataclass(frozen=True)
class IdaesUncertaintyResult(Record):
    parameters: tuple[IdaesUncertainParameter, ...]
    covariance: tuple[tuple[float, ...], ...]
    responses: tuple[IdaesResponse, ...]
    mode: Literal["fixed", "reoptimized"]
    method: Literal["implicit", "finite_difference"]
    execution: Literal["completed", "unresolved"]
    baseline: EvaluationResult
    values: tuple[float, ...] = ()
    jacobian: tuple[tuple[float, ...], ...] = ()
    steps: Mapping[str, float] = field(default_factory=dict)
    relative_tolerance: float = 1e-3
    absolute_tolerance: float = 1e-6
    perturbations: tuple[IdaesPerturbation, ...] = ()
    numerical: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    scope: Mapping[str, Any] = field(default_factory=lambda: _SCOPE)

    def _validate(self):
        if dict(self.scope) != _SCOPE:
            raise ValueError("Local uncertainty scope qualifications must be retained")
        if not self.parameters or not self.responses:
            raise ValueError("Supply parameters and selected responses")
        unique(tuple(p.name for p in self.parameters), "parameter names")
        unique(tuple(p.component for p in self.parameters), "parameter components")
        unique(tuple(r.name for r in self.responses), "response names")
        covariance_matrix(self.covariance, len(self.parameters))
        if not 0 < self.relative_tolerance < 1:
            raise ValueError("relative_tolerance must lie between zero and one")
        if self.absolute_tolerance < 0:
            raise ValueError("absolute_tolerance must be nonnegative")
        if (self.mode == "fixed") != (self.method == "implicit"):
            raise ValueError("Method disagrees with response mode")
        if self.mode == "fixed" and (self.steps or self.perturbations):
            raise ValueError("Implicit derivatives do not use perturbation steps")
        if self.mode == "reoptimized":
            if set(self.steps) != {p.name for p in self.parameters} or any(
                finite(v, "step") <= 0 for v in self.steps.values()
            ):
                raise ValueError("Positive steps must cover every parameter")
        if self.values and len(self.values) != len(self.responses):
            raise ValueError("Baseline values have the wrong shape")
        self._validate_baseline()
        if self.execution == "completed":
            if self.baseline.payload.feasibility != "feasible" or not self.values:
                raise ValueError("Completed propagation requires a checked baseline")
            if len(self.jacobian) != len(self.responses) or any(
                len(row) != len(self.parameters) for row in self.jacobian
            ):
                raise ValueError("Response Jacobian has the wrong shape")
            if self.mode == "reoptimized":
                expected = tuple(
                    (p.name, factor * self.steps[p.name])
                    for p in self.parameters
                    for factor in (-1, 1, -0.5, 0.5)
                )
                if tuple(
                    (p.parameter, p.offset) for p in self.perturbations
                ) != expected or not all(p.checked for p in self.perturbations):
                    raise ValueError(
                        "Reoptimized propagation requires all symmetric checks"
                    )
        elif self.jacobian or not self.diagnostics:
            raise ValueError("Unresolved propagation needs diagnostics and no Jacobian")

    def _validate_baseline(self):
        contract = self.baseline.contract
        bindings = {
            b["name"]: (b, kind)
            for kind in ("output", "control")
            for b in contract.model[f"{kind}s"]
        }
        constraints = {c.constraint_id: c for c in contract.constraints}
        quantities = {q.quantity_id: q for q in contract.quantities}
        checks = {c.constraint_ref: c for c in self.baseline.payload.constraint_checks}
        values = {v.quantity_ref: v.value for v in self.baseline.payload.values}
        rules = contract.operating["recourse_policy"]["rules"]
        if any(
            (r["fixed_value"] is None) != (self.mode == "reoptimized") for r in rules
        ):
            raise ValueError("Response mode disagrees with baseline actuator policy")
        for index, response in enumerate(self.responses):
            if response.kind == "margin":
                name = response.name.removeprefix("margin:")
                spec = constraints.get(name)
                if (
                    not response.name.startswith("margin:")
                    or spec is None
                    or spec.relation == "eq"
                    or quantities[spec.residual_ref].unit != response.unit
                ):
                    raise ValueError("Margin does not match a source inequality")
                expected = (
                    -checks[name].residual.value
                    if name in checks and checks[name].residual is not None
                    else None
                )
            else:
                binding, kind = bindings.get(response.name, ({}, None))
                if (
                    binding.get("component") != response.component
                    or binding.get("unit") != response.unit
                    or kind != response.kind
                ):
                    raise ValueError(
                        "Response does not match a declared output or control"
                    )
                expected = values.get(response.name)
            if self.values and self.values[index] != expected:
                raise ValueError("Response value differs from the checked baseline")

    @cached_property
    def statistics(self):
        """Compute J Sigma J.T and signed, correlation-aware variance allocations.

        The allocation J[j] * (Sigma @ J)[j] sums to variance, but can be
        negative. It is an accounting convention, not a causal attribution.
        """
        if self.execution != "completed":
            raise ValueError("Statistics require completed local propagation")
        j = np.asarray(self.jacobian)
        sigma = covariance_matrix(self.covariance, len(self.parameters))
        propagated = j @ sigma @ j.T
        if not np.all(np.isfinite(propagated)):
            raise ValueError("Nonfinite propagated covariance")
        rows = []
        for index, response in enumerate(self.responses):
            variance = max(0.0, float(propagated[index, index]))
            sd = float(np.sqrt(variance))
            contributions = j[index] * (sigma @ j[index])
            rows.append(
                {
                    "name": response.name,
                    "unit": response.unit,
                    "value": self.values[index],
                    "variance": variance,
                    "standard_deviation": sd,
                    "margin_in_standard_deviations": self.values[index] / sd
                    if response.kind == "margin" and sd
                    else None,
                    "parameter_contributions": dict(
                        zip((p.name for p in self.parameters), contributions.tolist())
                    ),
                    "ranking": tuple(
                        self.parameters[k].name
                        for k in np.argsort(-np.abs(contributions), kind="stable")
                    ),
                }
            )
        return freeze(
            {
                "responses": rows,
                "covariance": propagated.tolist(),
                "attribution": "Signed allocation J[j] * (Sigma @ J)[j]; cross-covariance shared equally between parameters. Negative allocations are possible; not causal or unique.",
            }
        )

    @property
    def ref(self):
        return reference(self.to_dict(), "idaes_local_uncertainty/v1")

    @property
    def result_id(self):
        return self.ref.artifact_id

    def to_json(self):
        return json.dumps(
            {
                "schema_version": self.ref.schema_version,
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
            or data.pop("schema_version", None) != "idaes_local_uncertainty/v1"
        ):
            raise ValueError("Unsupported local uncertainty schema")
        expected = data.pop("result_id", None)
        try:
            result = cls(**data)
        except (TypeError, KeyError) as exc:
            raise ValueError("Invalid local uncertainty fields") from exc
        if result.result_id != expected:
            raise ValueError("Local uncertainty integrity check failed")
        return result
