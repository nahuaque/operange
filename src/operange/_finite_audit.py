"""Shared enumeration and coverage bookkeeping for finite model adapters."""

from .contract_types import (
    Coverage,
    Diagnostic,
    Evidence,
    Measurement,
    RobustnessPayload,
)
from .engineering_results import RobustnessResult


def audit_finite(
    claim,
    evaluate,
    failure,
    *,
    method,
    scope=None,
    coverage_details=None,
    failure_constraints=None,
    diagnostic="A declared scenario could not be evaluated; complete coverage is unavailable.",
):
    """Keep model-specific failure proofs separate from enumeration mechanics.

    Every scenario identity is retained even when several share a realization.
    A verified failure survives unresolved evaluations of other scenarios.
    """
    evaluations, scenarios, unresolved, evidence = {}, [], [], []
    scope = {} if scope is None else scope
    contract = claim.contract
    for scenario in claim.domain.scenarios:
        result = evaluate(claim, scenario.values, contract=contract)
        evaluations[result.result_id] = result
        scenarios.append(
            {"name": scenario.name, "evaluation_ref": result.ref.to_dict()}
        )
        if result.execution != "completed":
            unresolved.append(scenario.name)
    witness = None
    for result in evaluations.values():
        affected = (
            failure_constraints(claim, result)
            if failure_constraints is not None
            else tuple(
                c.constraint_ref
                for c in result.payload.constraint_checks
                if c.assessment == "violated" and c.constraint_ref in claim.requirements
            )
        )
        if (
            result.execution == "completed"
            and result.payload.membership.status == "inside"
            and affected
        ):
            witness, proofs = failure(claim, result, affected)
            evidence.extend(proofs)
            break
    complete = not unresolved
    if complete:
        evidence.append(
            Evidence(
                "coverage",
                "domain_coverage",
                method,
                "verified",
                measurements=(Measurement("scenario_count", len(scenarios), "count"),),
                details=scope if coverage_details is None else coverage_details,
            )
        )
    return RobustnessResult(
        contract,
        {"query": "audit"},
        "completed" if complete else "unresolved",
        RobustnessPayload(
            "fail" if witness else "pass" if complete else "inconclusive",
            Coverage(
                "complete_finite" if complete else "partial",
                evaluated_support={"scenarios": scenarios, **scope},
                unexplored_support={"unresolved_scenarios": unresolved},
                evidence_refs=("coverage",) if complete else (),
            ),
            tuple(r.ref for r in evaluations.values()),
            witness=witness,
        ),
        tuple(evidence),
        tuple(
            Diagnostic("scenario_unresolved", name, diagnostic) for name in unresolved
        ),
        tuple(evaluations.values()),
    )
