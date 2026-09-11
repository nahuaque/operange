"""The common engineering claim interface and model-adapter capability boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from .contract_types import (
    Coverage,
    Diagnostic,
    EvaluationPayload,
    Membership,
    ProcessContract,
    Record,
    RobustnessPayload,
    SensitivityPayload,
    freeze,
    snapshot,
    unique,
)
from .distance import Distance
from .domains import ParameterSpace, UncertaintySet
from .engineering_results import EvaluationResult, RobustnessResult, SensitivityResult
from .recourse import RecoursePolicy


@dataclass(frozen=True)
class Capability(Record):
    supported: bool
    scope: str

    def _validate(self):
        if not self.scope.strip():
            raise ValueError("capability needs its scope or reason for rejection")


@dataclass(frozen=True)
class AdapterCapabilities(Record):
    evaluation: Capability
    sensitivity: Capability
    audit: Capability
    boundary: Capability
    breaking: Capability


@runtime_checkable
class ModelAdapter(Protocol):
    @property
    def base_contract(self) -> ProcessContract: ...

    @property
    def input_space(self) -> ParameterSpace: ...

    def describe(self, claim: Claim) -> ProcessContract: ...

    def capabilities(self, claim: Claim) -> AdapterCapabilities: ...

    def run(self, claim: Claim, operation: str, realization, options: dict): ...


def bind_contract(contract, domain, recourse, requirements=None):
    """Bind declarations to a portable result manifest, without serializing code."""
    refs = tuple(
        sorted(
            requirements
            if requirements is not None
            else (
                c.constraint_id for c in contract.constraints if c.kind == "requirement"
            )
        )
    )
    return replace(
        contract,
        domain=domain.to_manifest(),
        operating={
            **dict(contract.operating),
            "recourse_policy": recourse.to_dict(),
            "requirement_refs": refs,
        },
    )


def rejected_result(
    contract,
    operation,
    request,
    message,
    *,
    code="unsupported_capability",
    execution="unsupported",
):
    diagnostic = (Diagnostic(code, operation, message),)
    if operation == "evaluation":
        return EvaluationResult(
            contract,
            snapshot(request),
            execution,
            EvaluationPayload("not_evaluated", Membership("not_checked")),
            diagnostics=diagnostic,
        )
    if operation == "sensitivity":
        return SensitivityResult(
            contract,
            snapshot(request),
            execution,
            SensitivityPayload("not_evaluated"),
            diagnostics=diagnostic,
        )
    return RobustnessResult(
        contract,
        snapshot(request),
        execution,
        RobustnessPayload("not_assessed", Coverage("none")),
        diagnostics=diagnostic,
    )


@dataclass(frozen=True)
class Claim:
    """Bind an engineering model, domain, requirements and operating permissions.

    Requirements refer to declared engineering constraints in the adapter's
    registry. Supporting a domain's membership operation does not grant the
    adapter an optimization or universal-certification capability over it.
    """

    adapter: ModelAdapter
    domain: UncertaintySet
    recourse: RecoursePolicy
    requirements: tuple[str, ...] | None = None
    distance: Distance | None = None

    def __post_init__(self):
        if not isinstance(self.adapter, ModelAdapter):
            raise ValueError("adapter must implement ModelAdapter")
        if not isinstance(self.domain, UncertaintySet):
            raise ValueError("domain must implement UncertaintySet")
        if not isinstance(self.recourse, RecoursePolicy):
            raise ValueError("recourse must be a RecoursePolicy")
        self.adapter.input_space.check_alignment(self.domain.space)
        if self.distance is not None:
            if not isinstance(self.distance, Distance):
                raise ValueError("distance must implement Distance")
            self.domain.space.check_alignment(self.distance.space)
        base = self.adapter.base_contract
        requirements = (
            tuple(
                sorted(
                    c.constraint_id for c in base.constraints if c.kind == "requirement"
                )
            )
            if self.requirements is None
            else tuple(self.requirements)
        )
        if not requirements or any(not isinstance(r, str) for r in requirements):
            raise ValueError("claim needs named engineering requirements")
        unique(requirements, "claim requirements")
        known = {c.constraint_id for c in base.constraints if c.kind == "requirement"}
        if any(r not in known for r in requirements):
            raise ValueError("claim references an undeclared engineering requirement")
        object.__setattr__(self, "requirements", tuple(sorted(requirements)))
        for rule in self.recourse.rules:
            quantity = base.quantity(rule.control)
            if quantity.role != "control" or quantity.unit != rule.unit:
                raise ValueError(
                    "recourse control role or units do not match the model"
                )

    @property
    def contract(self):
        contract = self.adapter.describe(self)
        if (
            not isinstance(contract, ProcessContract)
            or contract.domain != freeze(self.domain.to_manifest())
            or contract.operating.get("recourse_policy")
            != freeze(self.recourse.to_dict())
            or contract.operating.get("requirement_refs") != self.requirements
        ):
            raise ValueError("adapter description does not bind the requested claim")
        return contract

    @property
    def capabilities(self):
        capabilities = self.adapter.capabilities(self)
        if not isinstance(capabilities, AdapterCapabilities):
            raise ValueError("adapter must declare typed capabilities")
        return capabilities

    def _run(self, operation, realization=None, **options):
        contract = self.contract
        capability = getattr(self.capabilities, operation)
        request = {
            "query": operation,
            "realization": snapshot(realization),
            **snapshot(options),
        }
        if operation in ("boundary", "breaking"):
            request["distance"] = self.distance.to_manifest() if self.distance else None
        if not capability.supported:
            return rejected_result(contract, operation, request, capability.scope)
        result = self.adapter.run(self, operation, realization, options)
        expected = (
            EvaluationResult
            if operation == "evaluation"
            else SensitivityResult
            if operation == "sensitivity"
            else RobustnessResult
        )
        if not isinstance(result, expected) or result.contract_ref != contract.ref:
            raise ValueError(
                "adapter returned a result for a different contract or query family"
            )
        return result

    def evaluate_result(self, realization=None, **options):
        return self._run("evaluation", realization, **options)

    def sensitivity_result(self, realization=None, **options):
        return self._run("sensitivity", realization, **options)

    def audit_result(self, **options):
        return self._run("audit", **options)

    def boundary_result(self, **options):
        return self._run("boundary", **options)

    def breaking_result(self, **options):
        return self._run("breaking", **options)
