"""Portable utility-target evidence; never an installed-network certificate."""

from fractions import Fraction

from ._heat_cascade import cascade, number, stream_conditions, upper
from .claim import rejected_result
from .contract_types import (
    ConstraintCheck,
    Coverage,
    Diagnostic,
    Evidence,
    EvaluationPayload,
    Measurement,
    Membership,
    QuantityValue,
    RobustnessPayload,
    Witness,
    snapshot,
)
from .engineering_results import EvaluationResult, RobustnessResult


def evaluate(claim, realization):
    model, contract = claim.adapter, claim.contract
    request = snapshot({"query": "evaluation", "realization": realization})
    try:
        point = claim.domain.space.validate(realization)
        check = claim.domain.membership(point)
    except (ValueError, TypeError, KeyError) as exc:
        return rejected_result(
            contract,
            "evaluation",
            request,
            str(exc),
            execution="invalid",
            code="invalid_realization",
        )
    evidence = [
        Evidence(
            "membership",
            "domain_membership",
            "finite_enumeration",
            "unresolved" if check.status == "unknown" else "verified",
            details=check.to_dict(),
        )
    ]
    membership = Membership(
        check.status, evidence_refs=() if check.status == "unknown" else ("membership",)
    )
    if check.status != "inside":
        return EvaluationResult(
            contract,
            request,
            "unresolved" if check.status == "unknown" else "invalid",
            EvaluationPayload(
                "unknown" if check.status == "unknown" else "not_evaluated", membership
            ),
            tuple(evidence),
            (
                Diagnostic(
                    "domain_membership",
                    "realization",
                    f"Stream-condition membership is {check.status}.",
                ),
            ),
        )
    try:
        conditions = stream_conditions(model, point)
    except ValueError as exc:
        return EvaluationResult(
            contract,
            request,
            "invalid",
            EvaluationPayload("not_evaluated", membership),
            tuple(evidence),
            (Diagnostic("invalid_stream_conditions", "streams", str(exc)),),
        )
    try:
        targets, details = cascade(model, conditions)
        values = {
            **point,
            **{k: number(v) for k, v in targets.items()},
            "energy_balance_residual": 0,
        }
        residuals = {
            f"{side}_utility_target": (
                f"{side}_utility_residual",
                upper(
                    targets[f"minimum_{side}_utility"]
                    - Fraction(getattr(model, f"{side}_utility_capacity_kw"))
                ),
            )
            for side in ("hot", "cold")
        }
    except (ArithmeticError, ValueError) as exc:
        return EvaluationResult(
            contract,
            request,
            "unresolved",
            EvaluationPayload("unknown", membership),
            tuple(evidence),
            (Diagnostic("heat_cascade_arithmetic_unresolved", "response", str(exc)),),
        )
    evidence.append(
        Evidence(
            "cascade",
            "thermodynamic_utility_targets",
            "exact_sensible_heat_cascade",
            "verified",
            assumptions=(
                "steady state; constant nonnegative heat-capacity flow; no phase changes",
                "fixed delta-T-min; unrestricted heat matches and splitting per scenario",
                "utilities at adequate temperatures; no heat pumps, storage or heat losses",
                "target feasibility does not establish installed-network feasibility",
            ),
            details=details,
        )
    )
    checks = [
        ConstraintCheck(
            "heat_balance",
            "satisfied",
            QuantityValue("energy_balance_residual", 0),
            ("cascade",),
        )
    ]
    for name, (residual_name, residual) in residuals.items():
        if name in claim.requirements:
            values[residual_name] = residual
            checks.append(
                ConstraintCheck(
                    name,
                    "satisfied" if residual <= model.tolerance_kw else "violated",
                    QuantityValue(residual_name, residual),
                    ("cascade",),
                )
            )
    return EvaluationResult(
        contract,
        request,
        "completed",
        EvaluationPayload(
            "feasible",
            membership,
            tuple(QuantityValue(k, v) for k, v in values.items()),
            tuple(checks),
            included_constraints=("heat_balance",),
        ),
        tuple(evidence),
    )


def audit(claim):
    evaluations, scenarios, unresolved, evidence = {}, [], [], []
    for scenario in claim.domain.scenarios:
        result = evaluate(claim, scenario.values)
        evaluations[result.result_id] = result
        scenarios.append(
            {"name": scenario.name, "evaluation_ref": result.ref.to_dict()}
        )
        if result.execution != "completed":
            unresolved.append(scenario.name)
    complete = not unresolved
    witness = None
    for result in evaluations.values():
        affected = tuple(
            c.constraint_ref
            for c in result.payload.constraint_checks
            if c.assessment == "violated"
        )
        if result.execution == "completed" and affected:
            evidence.extend(
                (
                    Evidence(
                        "witness_membership",
                        "domain_membership",
                        "finite_enumeration",
                        "verified",
                        details={"evaluation_ref": result.ref.to_dict()},
                    ),
                    Evidence(
                        "target_failure",
                        "recourse_infeasibility",
                        "minimum_utility_exceeds_capacity",
                        "verified",
                        details={
                            "evaluation_ref": result.ref.to_dict(),
                            "scope": "utility target exceeds a selected limit even with unrestricted heat matching",
                            "installed_network_feasibility": "not_assessed",
                        },
                    ),
                )
            )
            witness = Witness(
                "individual_infeasibility",
                (result.request["realization"],),
                affected,
                ("witness_membership",),
                ("target_failure",),
                details={
                    "evaluation_ref": result.ref.to_dict(),
                    "scope": "thermodynamic_utility_targets",
                },
            )
            break
    scope = {
        "scope": "thermodynamic_utility_targets",
        "installed_network_feasibility": "not_assessed",
        "continuous_uncertainty_coverage": False,
    }
    if complete:
        evidence.append(
            Evidence(
                "coverage",
                "domain_coverage",
                "finite_scenarios_with_exact_heat_cascades",
                "verified",
                measurements=(Measurement("scenario_count", len(scenarios), "count"),),
                details=scope,
            )
        )
    coverage = Coverage(
        "complete_finite" if complete else "partial",
        evaluated_support={"scenarios": scenarios, **scope},
        unexplored_support={"unresolved_scenarios": unresolved},
        evidence_refs=("coverage",) if complete else (),
    )
    return RobustnessResult(
        claim.contract,
        {"query": "audit"},
        "completed" if complete else "unresolved",
        RobustnessPayload(
            "fail" if witness else "pass" if complete else "inconclusive",
            coverage,
            tuple(r.ref for r in evaluations.values()),
            witness=witness,
        ),
        tuple(evidence),
        tuple(
            Diagnostic(
                "scenario_unresolved",
                name,
                "A declared stream scenario could not be evaluated; complete coverage is unavailable.",
            )
            for name in unresolved
        ),
        tuple(evaluations.values()),
    )
