"""Finite numerical evaluation; solver status never establishes infeasibility."""

from .._finite_audit import audit_finite
from ..claim import rejected_result
from ..contract_types import (
    ConstraintCheck,
    Diagnostic,
    EvaluationPayload,
    Evidence,
    Membership,
    QuantityValue,
    snapshot,
)
from ..engineering_results import EvaluationResult
from ._idaes_execution import OperatingPointRequest, execute_point


def evaluate_point(
    claim, realization=None, *, contract=None, candidate=None, objective=False
):
    """Bind membership and export evidence, retaining the checked runtime point."""
    adapter = claim.adapter
    contract = claim.contract if contract is None else contract
    request = {"query": "evaluation", "realization": snapshot(realization)}
    try:
        point = adapter.input_space.validate(
            adapter.input_space.nominal if realization is None else realization
        )
        membership = claim.domain.membership(point)
    except (ValueError, TypeError) as exc:
        return rejected_result(
            contract,
            "evaluation",
            request,
            str(exc),
            code="invalid_realization",
            execution="invalid",
        ), None
    evidence = [
        Evidence(
            "membership",
            "domain_membership",
            "declared_domain_membership",
            "unresolved" if membership.status == "unknown" else "verified",
            details={"check": membership.to_dict()},
        )
    ]
    member = Membership(
        membership.status,
        evidence_refs=("membership",) if membership.status != "unknown" else (),
    )
    if membership.status != "inside":
        return EvaluationResult(
            contract,
            request,
            "invalid" if membership.status == "outside" else "unresolved",
            EvaluationPayload(
                "not_evaluated" if membership.status == "outside" else "unknown", member
            ),
            tuple(evidence),
            (
                Diagnostic(
                    "domain_membership",
                    "realization",
                    f"Membership is {membership.status}",
                ),
            ),
        ), None
    execution = execute_point(
        adapter,
        OperatingPointRequest.from_claim(claim, point),
        candidate=candidate,
        objective=objective,
    )
    return result_from_execution(
        contract, request, member, evidence, execution
    ), execution


def result_from_execution(contract, request, member, evidence, execution):
    """Translate checked runtime facts into the existing portable schema."""
    adapter, checked, candidate = (
        execution.adapter,
        execution.checked,
        execution.candidate,
    )
    values = tuple(
        QuantityValue(name, value)
        for name, value in execution.request.realization.items()
    )
    checks = tuple(ConstraintCheck(row.name, "unknown") for row in execution.rows)
    evidence.extend(execution.before)
    if candidate is not None:
        evidence.append(
            Evidence(
                "solver",
                "solver_termination",
                "pyomo_solver",
                "verified" if candidate.solved else "unresolved",
                backend=adapter.solver,
                details=candidate.details,
            )
        )
    if checked is not None:
        values = tuple(
            QuantityValue(name, value) for name, value in checked.values.items()
        )
        evidence.append(
            Evidence(
                "residuals",
                "numerical_feasibility",
                "pyomo_expression_and_bound_checks",
                "verified",
                details={
                    "source_fingerprint": contract.model["source_fingerprint"],
                    "scope": "Floating-point substitution in all included source constraints and bounds; no global optimality or independent certificate.",
                    "residuals": {row.name: value for row, value in checked.residuals},
                },
            )
        )
        checks = tuple(
            ConstraintCheck(
                row.name,
                "satisfied"
                if (abs(value) if row.side == "eq" else value) <= row.tolerance
                else "violated",
                QuantityValue(row.residual, value),
                ("residuals",),
            )
            for row, value in checked.residuals
        )
        if checked.objective is not None:
            evidence.append(
                Evidence(
                    "source_objective",
                    "numerical_objective_value",
                    "pyomo_source_expression",
                    "verified",
                    details=checked.objective,
                )
            )
    evidence.extend(execution.after)
    return EvaluationResult(
        contract,
        request,
        "completed" if execution.feasible else "unresolved",
        EvaluationPayload(
            "feasible" if execution.feasible else "unknown",
            member,
            values,
            checks,
            included_constraints=tuple(row.name for row in execution.rows),
        ),
        tuple(evidence),
        () if execution.failure is None else (execution.failure,),
    )


def evaluate_result(claim, realization=None, *, contract=None):
    return evaluate_point(claim, realization, contract=contract)[0]


def evaluate_candidate(claim, realization, candidate, *, contract=None):
    """Check a joint solver candidate through the same execution boundary."""
    return evaluate_point(
        claim, realization, contract=contract, candidate=candidate, objective=True
    )[0]


def audit_result(claim):
    def no_failure(*args):
        raise AssertionError(
            "A numerical IDAES candidate cannot establish recourse infeasibility"
        )

    return audit_finite(
        claim,
        evaluate_result,
        no_failure,
        method="complete_finite_idaes_numerical_checks",
        scope={
            "coverage": "declared finite realizations only",
            "evidence": "numerical residual checks; not independently certified",
            "controls": "fixed as declared or independently adjustable after all inputs are observed",
        },
        diagnostic="A scenario lacks a checked numerical candidate; solver infeasibility is not a proof.",
    )
