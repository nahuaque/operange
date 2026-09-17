"""Portable evidence for finite shared-command optimization; no backend imports."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Any, Literal

from ..claim import bind_contract
from ..contract_types import Diagnostic, ProcessContract, Record, plain, reference
from ..domains import FiniteSet
from ..engineering_results import RobustnessResult, result_from_json
from ..primitives import finite
from ..recourse import DecisionRule, RecoursePolicy


def fixed_policy(controls, commands):
    return RecoursePolicy(
        "fixed",
        tuple(
            DecisionRule(
                c["name"], c["unit"], "operation", fixed_value=commands[c["name"]]
            )
            for c in controls
        ),
    )


_SCOPE = {
    "coverage": "declared_finite_scenarios_only",
    "controls": "One common command vector; no observed inputs or parameters.",
    "objective": "Worst-case source objective; no probabilities or expected cost inferred.",
    "checks": "Numerical substitution of scenario states and exact shared commands in separate source-model clones.",
    "optimality": "not_certified",
    "infeasibility": "not_certified",
    "confidence_level": None,
}


@dataclass(frozen=True)
class IdaesSharedControlResult(Record):
    """Checked common commands and finite coverage, or an unresolved search.

    ``commands`` may retain an unsuccessful candidate for inspection. Use
    ``execution == 'completed'`` before treating the plan as checked. Loading
    validates record consistency and identity, not the nonlinear mathematics.
    """

    contract: ProcessContract
    domain: FiniteSet
    objective: Mapping[str, Any]
    execution: Literal["completed", "unresolved"]
    commands: Mapping[str, float] = field(default_factory=dict)
    objective_value: float | None = None
    scenario_objectives: Mapping[str, float] = field(default_factory=dict)
    audit: RobustnessResult | None = None
    solver: Mapping[str, Any] = field(default_factory=dict)
    preparation: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    scope: Mapping[str, Any] = field(default_factory=lambda: _SCOPE)

    def _validate(self):
        if dict(self.scope) != _SCOPE:
            raise ValueError(
                "Shared-control scope must retain numerical finite qualifications"
            )
        controls = self.contract.model["controls"]
        if not controls:
            raise ValueError("Shared optimization needs at least one actuator")
        inputs = {b["name"]: b["unit"] for b in self.contract.model["inputs"]}
        if {c.name: c.unit for c in self.domain.space.coordinates} != inputs:
            raise ValueError("Shared-control domain differs from the source inputs")
        source = self.contract.model["source"]["objectives"]
        if (
            len(source) != 1
            or dict(self.objective)
            != {
                **dict(source[0]),
                "unit": self.objective.get("unit"),
                "aggregation": "worst_case",
            }
            or not isinstance(self.objective.get("unit"), str)
            or not self.objective["unit"].strip()
        ):
            raise ValueError(
                "Shared objective must identify the single source objective"
            )
        if self.objective["sense"] not in ("minimize", "maximize"):
            raise ValueError("Unsupported objective sense")
        if self.commands and set(self.commands) != {c["name"] for c in controls}:
            raise ValueError("Shared commands must cover every actuator")
        for name, value in self.commands.items():
            finite(value, f"command {name}")
        expected_values = {}
        if self.audit is not None:
            if not self.commands:
                raise ValueError("A shared-command audit needs commands")
            expected_contract = bind_contract(
                self.contract, self.domain, fixed_policy(controls, self.commands)
            )
            if self.audit.contract != expected_contract:
                raise ValueError(
                    "Audit changes the shared commands, domain or source model"
                )
            evaluations = {e.result_id: e for e in self.audit.supporting_evaluations}
            support = self.audit.payload.coverage.evaluated_support["scenarios"]
            if tuple(s["name"] for s in support) != tuple(
                s.name for s in self.domain.scenarios
            ):
                raise ValueError("Shared audit must retain every scenario label")
            for scenario, entry in zip(self.domain.scenarios, support):
                evaluation = evaluations[entry["evaluation_ref"]["artifact_id"]]
                if dict(evaluation.request["realization"]) != dict(scenario.values):
                    raise ValueError("Shared audit changes a scenario realization")
                if evaluation.payload.feasibility == "feasible":
                    values = {
                        v.quantity_ref: v.value for v in evaluation.payload.values
                    }
                    if any(
                        values.get(name) != value
                        for name, value in self.commands.items()
                    ):
                        raise ValueError(
                            "Scenario does not use the exact shared commands"
                        )
                    evidence = [
                        e
                        for e in evaluation.evidence
                        if e.evidence_id == "source_objective"
                    ]
                    if len(evidence) != 1 or evidence[0].outcome != "verified":
                        raise ValueError("Missing checked scenario objective")
                    details = evidence[0].details
                    if any(
                        details[k] != self.objective[k]
                        for k in ("component", "sense", "unit")
                    ):
                        raise ValueError(
                            "Scenario objective differs from the source objective"
                        )
                    expected_values[scenario.name] = details["value"]
        if dict(self.scenario_objectives) != expected_values:
            raise ValueError("Scenario objective values disagree with source checks")
        complete = self.audit is not None and self.audit.payload.verdict == "pass"
        if (self.execution == "completed") != complete:
            raise ValueError(
                "Shared optimization execution disagrees with finite audit"
            )
        if complete:
            aggregate = max if self.objective["sense"] == "minimize" else min
            if self.objective_value != aggregate(expected_values.values()):
                raise ValueError("Worst-case objective disagrees with scenario values")
        elif self.objective_value is not None or not self.diagnostics:
            raise ValueError(
                "Unresolved shared optimization needs diagnostics and no aggregate value"
            )

    def to_dict(self):
        data = plain(self)
        data["audit"] = None if self.audit is None else self.audit.to_dict(compact=True)
        return data

    @property
    def ref(self):
        return reference(self.to_dict(), "idaes_shared_controls/v1")

    @property
    def result_id(self):
        return self.ref.artifact_id

    def to_json(self):
        return json.dumps(
            {
                "schema_version": "idaes_shared_controls/v1",
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
            or data.pop("schema_version", None) != "idaes_shared_controls/v1"
        ):
            raise ValueError("Unsupported shared-control result schema")
        expected = data.pop("result_id", None)
        try:
            if data.get("audit") is not None:
                data["audit"] = result_from_json(json.dumps(data["audit"]))
            result = cls(**data)
        except (TypeError, KeyError) as exc:
            raise ValueError("Invalid shared-control result fields") from exc
        if result.result_id != expected:
            raise ValueError("Shared-control result integrity check failed")
        return result
