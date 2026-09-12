"""Versioned, self-contained engineering results; loading never executes a model."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from functools import cached_property
import json
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from .contract_types import (
    ArtifactRef,
    Diagnostic,
    Evidence,
    EvaluationPayload,
    ProcessContract,
    Record,
    RobustnessPayload,
    SensitivityPayload,
    plain,
    reference,
    unique,
)


def _records(value):
    if is_dataclass(value):
        yield value
        for item in fields(value):
            yield from _records(getattr(value, item.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _records(item)


@dataclass(frozen=True)
class _Result(Record):
    contract: ProcessContract
    request: Mapping[str, Any]
    execution: Literal["completed", "unresolved", "unsupported", "invalid"]
    payload: EvaluationPayload | SensitivityPayload | RobustnessPayload
    evidence: tuple[Evidence, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    supporting_evaluations: tuple[EvaluationResult, ...] = field(default=(), repr=False)
    schema_version: ClassVar[str] = "process_result/v1"
    kind: ClassVar[str]

    @property
    def contract_ref(self):
        return self.contract.ref

    def _metadata(self):
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "contract_ref": plain(self.contract_ref),
            "request": plain(self.request),
            "execution": self.execution,
            "payload": plain(self.payload),
            "evidence": plain(self.evidence),
            "diagnostics": plain(self.diagnostics),
        }

    def _body(self):
        return {
            **self._metadata(),
            "artifacts": {
                "contract": self.contract.to_dict(),
                "evaluations": [item.to_dict() for item in self.supporting_evaluations],
            },
        }

    @cached_property
    def ref(self):
        return reference(self._body(), self.schema_version)

    @property
    def result_id(self):
        return self.ref.artifact_id

    def to_dict(self, *, compact=False):
        if compact:
            return _compact_bundle(self)
        return {**self._body(), "result_id": self.result_id}

    def to_json(self, *, indent=2, compact=False):
        return json.dumps(
            self.to_dict(compact=compact),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, text):
        result = result_from_json(text)
        if not isinstance(result, cls):
            raise ValueError(f"expected {cls.__name__}")
        return result

    def _validate(self):
        unique(tuple(e.evidence_id for e in self.evidence), "evidence identities")
        proofs = {e.evidence_id: e for e in self.evidence}
        evaluations = {e.result_id: e for e in self.supporting_evaluations}
        if len(evaluations) != len(self.supporting_evaluations):
            raise ValueError("duplicate supporting evaluations")
        if any(e.contract_ref != self.contract_ref for e in evaluations.values()):
            raise ValueError("supporting evaluations use a different contract")
        for item in _records(self.payload):
            for name in (
                "evidence_refs",
                "membership_evidence_refs",
                "recourse_evidence_refs",
            ):
                for ref in getattr(item, name, ()):
                    if ref not in proofs or proofs[ref].outcome != "verified":
                        raise ValueError(f"missing or unverified evidence: {ref}")
            if isinstance(item, ArtifactRef):
                if (
                    item.artifact_id not in evaluations
                    or evaluations[item.artifact_id].ref != item
                ):
                    raise ValueError("unresolved evaluation reference")
        if self.execution in ("invalid", "unsupported"):
            if not self.diagnostics:
                raise ValueError("rejected requests need a diagnostic")
            state = (
                getattr(self.payload, "feasibility", None)
                or getattr(self.payload, "availability", None)
                or getattr(self.payload, "verdict", None)
            )
            if state not in ("not_evaluated", "not_assessed"):
                raise ValueError("rejected request cannot carry a semantic conclusion")
            if self.supporting_evaluations:
                raise ValueError("rejected request cannot carry executed evaluations")


@dataclass(frozen=True)
class EvaluationResult(_Result):
    payload: EvaluationPayload
    kind: ClassVar[str] = "evaluation"

    def _validate(self):
        super()._validate()
        if self.supporting_evaluations:
            raise ValueError("evaluation results cannot contain other evaluations")
        specs = {c.constraint_id: c for c in self.contract.constraints}
        checks = {c.constraint_ref: c for c in self.payload.constraint_checks}
        for value in self.payload.values:
            self.contract.check_value(value)
        values = {value.quantity_ref: value.value for value in self.payload.values}
        for check in checks.values():
            if check.constraint_ref not in specs:
                raise ValueError("unknown checked constraint")
            spec = specs[check.constraint_ref]
            if check.residual is not None:
                self.contract.check_value(check.residual)
                if check.residual.quantity_ref != spec.residual_ref:
                    raise ValueError("constraint check uses the wrong residual")
                if (
                    spec.residual_ref in values
                    and values[spec.residual_ref] != check.residual.value
                ):
                    raise ValueError(
                        "constraint residual contradicts stored quantity value"
                    )
                values[spec.residual_ref] = check.residual.value

                def flattened(value):
                    if isinstance(value, tuple):
                        for item in value:
                            yield from flattened(item)
                    else:
                        yield value

                satisfied = all(
                    (abs(v) if spec.relation == "eq" else v) <= spec.tolerance
                    for v in flattened(check.residual.value)
                )
                if (
                    check.assessment in ("satisfied", "violated")
                    and (check.assessment == "satisfied") != satisfied
                ):
                    raise ValueError("constraint assessment contradicts residual")
            if (
                check.assessment in ("satisfied", "violated")
                and not check.evidence_refs
            ):
                raise ValueError("constraint conclusion needs evidence")
        included = self.payload.included_constraints
        if self.payload.feasibility == "not_evaluated" and (
            self.payload.values
            or self.payload.constraint_checks
            or self.payload.objective
        ):
            raise ValueError("unevaluated result cannot carry an operating solution")
        if any(c not in specs for c in included):
            raise ValueError("unknown feasibility constraint")
        if self.payload.feasibility == "feasible":
            if not included or any(
                c not in checks or checks[c].assessment != "satisfied" for c in included
            ):
                raise ValueError(
                    "feasibility requires all included constraints to be satisfied"
                )
        if self.payload.feasibility == "infeasible" and not any(
            e.subject == "recourse_infeasibility" and e.outcome == "verified"
            for e in self.evidence
        ):
            raise ValueError("infeasibility needs recourse infeasibility evidence")
        if (
            self.payload.membership.status in ("inside", "outside")
            and not self.payload.membership.evidence_refs
        ):
            raise ValueError("membership conclusion needs evidence")
        obj = self.payload.objective
        if obj:
            if (
                obj.attained_value is not None
                and obj.quantity_ref in values
                and values[obj.quantity_ref] != obj.attained_value
            ):
                raise ValueError("attained objective contradicts stored quantity value")
            if self.payload.feasibility != "feasible":
                raise ValueError("objective requires a verified feasible response")
            spec = self.contract.quantity(obj.quantity_ref)
            if spec.axes or obj.lower.unit != spec.unit:
                raise ValueError("objective must be scalar with declared units")
            tol = self.contract.numerical_policy.get("absolute_tolerance", 0)
            if obj.attained_value is not None:
                if (
                    obj.lower.value is not None
                    and obj.attained_value < obj.lower.value - tol
                ):
                    raise ValueError("objective below lower bound")
                if (
                    obj.upper.value is not None
                    and obj.attained_value > obj.upper.value + tol
                ):
                    raise ValueError("objective above upper bound")
            if obj.optimality == "verified" and (
                obj.lower.value is None
                or obj.upper.value is None
                or obj.upper.value - obj.lower.value > tol
            ):
                raise ValueError("verified optimality requires closed bounds")


@dataclass(frozen=True)
class SensitivityResult(_Result):
    payload: SensitivityPayload
    kind: ClassVar[str] = "sensitivity"

    def _validate(self):
        super()._validate()
        derivative = self.payload.derivative
        if derivative:
            if self.execution != "completed":
                raise ValueError("available derivative requires completed execution")
            base = next(
                e
                for e in self.supporting_evaluations
                if e.ref == self.payload.base_evaluation_ref
            )
            if base.payload.feasibility != "feasible":
                raise ValueError("available derivative requires a feasible base")
            if not any(
                e.subject == "derivative" and e.outcome == "verified"
                for e in self.evidence
            ):
                raise ValueError("derivative requires evidence")
            for name, unit in zip(
                derivative.output_coordinates, derivative.output_units
            ):
                spec = self.contract.quantity(name)
                if spec.axes or spec.unit != unit:
                    raise ValueError(
                        "output derivative coordinate has wrong unit/shape"
                    )
            for name, unit in zip(derivative.input_coordinates, derivative.input_units):
                spec = self.contract.quantity(name)
                expected = (
                    spec.unit if derivative.coordinate_space == "physical" else "1"
                )
                if spec.role != "input" or spec.axes or unit != expected:
                    raise ValueError(
                        "input derivative coordinate has wrong role/unit/shape"
                    )


@dataclass(frozen=True)
class RobustnessResult(_Result):
    payload: RobustnessPayload
    kind: ClassVar[str] = "robustness"

    def _validate(self):
        super()._validate()
        witness = self.payload.witness
        constraints = {c.constraint_id for c in self.contract.constraints}
        if witness and (
            not witness.affected_constraints
            or any(c not in constraints for c in witness.affected_constraints)
        ):
            raise ValueError("witness needs declared affected constraints")
        proofs = {e.evidence_id: e for e in self.evidence}
        if witness:
            if any(
                proofs[r].subject != "domain_membership"
                for r in witness.membership_evidence_refs
            ) or any(
                proofs[r].subject != "recourse_infeasibility"
                for r in witness.recourse_evidence_refs
            ):
                raise ValueError("witness references the wrong proof obligations")
        if self.payload.verdict == "pass":
            evaluations = self.supporting_evaluations
            passing = [
                e.payload.feasibility == "feasible"
                and all(
                    c.assessment == "satisfied" for c in e.payload.constraint_checks
                )
                for e in evaluations
            ]
            if not any(passing) or (
                self.payload.coverage.method == "complete_finite" and not all(passing)
            ):
                raise ValueError("passing audit requires verified supporting responses")
        search = self.payload.search
        if search:
            self.contract.check_value(search.threshold)
            if (
                search.resolution in ("minimum_verified", "unreachable")
                and not search.evidence_refs
            ):
                raise ValueError("resolved search requires evidence")


def _compact_bundle(result):
    """Deduplicate declarations without changing any process_result/v1 identity."""
    contracts = {}

    def entry(item):
        identity = item.contract_ref.artifact_id
        if identity not in contracts:
            contracts[identity] = item.contract.to_dict()
        return {
            **item._metadata(),
            "result_id": item.result_id,
            "artifacts": {
                "contract": identity,
                "evaluations": [e.result_id for e in item.supporting_evaluations],
            },
        }

    root = entry(result)
    evaluations = {e.result_id: entry(e) for e in result.supporting_evaluations}
    return {
        "schema_version": "process_result_bundle/v1",
        "result": root,
        "contracts": contracts,
        "evaluations": evaluations,
    }


def _expand_bundle(data):
    if set(data) != {"schema_version", "result", "contracts", "evaluations"}:
        raise ValueError("invalid compact bundle fields")
    if not isinstance(data["contracts"], dict) or not isinstance(
        data["evaluations"], dict
    ):
        raise ValueError("invalid compact artifact tables")
    contracts, used_evaluations = {}, set()

    def expand(item, *, root=False):
        if not isinstance(item, dict):
            raise ValueError("invalid compact result")
        refs = item.get("artifacts")
        if not isinstance(refs, dict) or set(refs) != {"contract", "evaluations"}:
            raise ValueError("invalid compact artifact references")
        key, evaluations = refs["contract"], refs["evaluations"]
        if not isinstance(key, str) or key not in data["contracts"]:
            raise ValueError("unresolved contract reference")
        if (
            not isinstance(evaluations, list)
            or any(not isinstance(e, str) for e in evaluations)
            or (not root and evaluations)
        ):
            raise ValueError("invalid compact evaluation references")
        unique(tuple(evaluations), "evaluation references")
        if key not in contracts:
            raw = data["contracts"][key]
            if not isinstance(raw, dict):
                raise ValueError("invalid compact contract")
            try:
                contracts[key] = ProcessContract(**raw)
            except TypeError as exc:
                raise ValueError("invalid compact contract fields") from exc
            if contracts[key].ref.artifact_id != key:
                raise ValueError("contract artifact digest mismatch")
        expanded = []
        for identity in evaluations:
            evaluation = data["evaluations"].get(identity)
            if (
                not isinstance(evaluation, dict)
                or evaluation.get("result_id") != identity
            ):
                raise ValueError("unresolved evaluation reference")
            if evaluation.get("kind") != "evaluation":
                raise ValueError("supporting artifact must be an evaluation")
            used_evaluations.add(identity)
            expanded.append(expand(evaluation))
        return {
            **item,
            "artifacts": {"contract": contracts[key], "evaluations": expanded},
        }

    expanded = expand(data["result"], root=True)
    if set(contracts) != set(data["contracts"]) or used_evaluations != set(
        data["evaluations"]
    ):
        raise ValueError("unreferenced compact artifacts")
    return expanded


def _decode(data):
    expected = {
        "schema_version",
        "kind",
        "result_id",
        "contract_ref",
        "request",
        "execution",
        "payload",
        "evidence",
        "diagnostics",
        "artifacts",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("invalid result envelope fields")
    if data["schema_version"] != "process_result/v1":
        raise ValueError("unsupported result schema")
    classes = {
        "evaluation": EvaluationResult,
        "sensitivity": SensitivityResult,
        "robustness": RobustnessResult,
    }
    if not isinstance(data["kind"], str) or data["kind"] not in classes:
        raise ValueError("unsupported result kind")
    artifacts = data["artifacts"]
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != {"contract", "evaluations"}
        or not isinstance(artifacts["evaluations"], list)
    ):
        raise ValueError("invalid artifact bundle")
    try:
        result = classes[data["kind"]](
            contract=artifacts["contract"],
            request=data["request"],
            execution=data["execution"],
            payload=data["payload"],
            evidence=data["evidence"],
            diagnostics=data["diagnostics"],
            supporting_evaluations=tuple(
                _decode(item) for item in artifacts["evaluations"]
            ),
        )
        contract_ref = ArtifactRef(**data["contract_ref"])
    except (TypeError, KeyError) as exc:
        raise ValueError("invalid result fields") from exc
    if result.contract_ref != contract_ref or result.result_id != data["result_id"]:
        raise ValueError("artifact digest mismatch")
    return result


def result_from_json(
    text: str,
) -> EvaluationResult | SensitivityResult | RobustnessResult:
    """Validate a portable bundle. Evidence is recorded, not re-proved on load.

    Use ``verify_result`` for independent checks of supported certificates.
    """

    def pairs(items):
        unique(tuple(k for k, _ in items), "JSON keys")
        return dict(items)

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    data = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if (
        isinstance(data, dict)
        and data.get("schema_version") == "process_result_bundle/v1"
    ):
        data = _expand_bundle(data)
    return _decode(data)
