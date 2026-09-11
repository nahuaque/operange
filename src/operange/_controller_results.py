"""Execute prescribed commands and retain physical failures without redispatch."""

from ._finite_audit import audit_finite
from ._linear_process_results import _included, _physical_response
from .claim import rejected_result
from .contract_types import (
    ConstraintCheck,
    Diagnostic,
    Evidence,
    EvaluationPayload,
    Membership,
    QuantityValue,
    Witness,
    snapshot,
)
from .engineering_results import EvaluationResult


def evaluate_result(claim, realization, *, contract=None):
    contract = claim.contract if contract is None else contract
    request = snapshot({"query": "evaluation", "realization": realization})
    try:
        check = claim.domain.membership(realization)
        point = claim.domain.space.validate(realization)
    except (ValueError, TypeError, KeyError) as exc:
        return rejected_result(
            contract,
            "evaluation",
            request,
            str(exc),
            code="invalid_realization",
            execution="invalid",
        )
    evidence = [
        Evidence(
            "membership",
            "domain_membership",
            "declared_domain_membership",
            "unresolved" if check.status == "unknown" else "verified",
            details={"check": check.to_dict()},
        )
    ]
    member = Membership(
        check.status, evidence_refs=("membership",) if check.status != "unknown" else ()
    )
    if check.status != "inside":
        return EvaluationResult(
            contract,
            request,
            "invalid" if check.status == "outside" else "unresolved",
            EvaluationPayload(
                "not_evaluated" if check.status == "outside" else "unknown", member
            ),
            tuple(evidence),
            (
                Diagnostic(
                    "domain_membership",
                    "realization",
                    f"Domain membership is {check.status}.",
                ),
            ),
        )
    included = _included(claim, contract)
    values = [QuantityValue(n, v) for n, v in point.items()]
    checks = tuple(ConstraintCheck(c.constraint_id, "unknown") for c in included)
    feasibility, message = "unknown", "Controller response was not resolved."
    controller = claim.adapter.controller
    try:
        commands = controller.commands(point)
        values.extend(QuantityValue(n, v) for n, v in commands.items())
        evidence.append(
            Evidence(
                "controller",
                "controller_execution",
                "exact_affine_command_evaluation",
                "verified",
                details={
                    "controller_ref": controller.ref.to_dict(),
                    "commands": commands,
                    "observations_by_control": {
                        r.control: {n: point[n] for n in r.observes}
                        for r in controller.rules
                    },
                    "arithmetic": "exact rational affine sum followed by one rounding to a physical float command",
                    "clipping": False,
                },
            )
        )
        physical, residuals = _physical_response(
            claim.adapter.model, claim.requirements, point, commands
        )
        violated = tuple(
            c.constraint_id
            for c in included
            if (
                abs(residuals[c.constraint_id])
                if c.relation == "eq"
                else residuals[c.constraint_id]
            )
            > c.tolerance
        )
        evidence.append(
            Evidence(
                "response",
                "controller_response",
                "exact_physical_residual_check",
                "verified",
                details={
                    "controller_ref": controller.ref.to_dict(),
                    "arithmetic": "exact physical residuals rounded upward for export",
                    "scope": "physical checks of the prescribed float commands only",
                },
            )
        )
        values = [QuantityValue(n, v) for n, v in physical.items()]
        checks = tuple(
            ConstraintCheck(
                c.constraint_id,
                "violated" if c.constraint_id in violated else "satisfied",
                QuantityValue(c.residual_ref, residuals[c.constraint_id]),
                ("response",),
            )
            for c in included
        )
        feasibility = "infeasible" if violated else "feasible"
        if violated:
            evidence.append(
                Evidence(
                    "recourse",
                    "recourse_infeasibility",
                    "fixed_controller_violation",
                    "verified",
                    details={
                        "controller_ref": controller.ref.to_dict(),
                        "commands": commands,
                        "violated_constraints": violated,
                        "scope": "these uniquely prescribed commands violate included constraints; other controllers not assessed",
                    },
                )
            )
    except (ValueError, OverflowError) as exc:
        message = str(exc)
    return EvaluationResult(
        contract,
        request,
        "unresolved" if feasibility == "unknown" else "completed",
        EvaluationPayload(
            feasibility,
            member,
            tuple(values),
            checks,
            included_constraints=tuple(c.constraint_id for c in included),
        ),
        tuple(evidence),
        (Diagnostic("controller_response_unresolved", "response", message),)
        if feasibility == "unknown"
        else (),
    )


def _failed_constraints(claim, result):
    return tuple(
        c.constraint_ref
        for c in result.payload.constraint_checks
        if c.assessment == "violated"
    )


def _failure(claim, result, affected):
    return Witness(
        "fixed_policy_failure",
        (result.request["realization"],),
        affected,
        ("membership",),
        ("recourse",),
        details={
            "controller_ref": claim.adapter.controller.ref.to_dict(),
            "evaluation_ref": result.ref.to_dict(),
            "scope": "failure of this frozen controller; other controllers not assessed",
        },
    ), tuple(
        e
        for e in result.evidence
        if e.evidence_id in ("membership", "controller", "recourse")
    )


def audit_result(claim):
    return audit_finite(
        claim,
        evaluate_result,
        _failure,
        method="complete_finite_controller_replay",
        scope={
            "controller_ref": claim.adapter.controller.ref.to_dict(),
            "controls": "prescribed by the same frozen rule at every scenario",
            "coverage": "declared finite realizations only; no retuning or redispatch",
        },
        failure_constraints=_failed_constraints,
        diagnostic="A scenario could not be evaluated under the frozen controller.",
    )
