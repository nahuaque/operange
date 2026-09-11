"""Exact finite-scenario/time-segment evidence for supplied startup profiles."""

from fractions import Fraction
from math import inf, nextafter

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
from .primitives import finite


def _float(value):
    return finite(float(value), "startup response")


def _upper(value):
    """Round a rational residual toward +infinity before comparing tolerance."""
    number = _float(value)
    return finite(
        nextafter(number, inf) if Fraction(number) < value else number, "residual bound"
    )


def _transforms(model, point):
    transformed = []
    horizon = Fraction(model.horizon_seconds)
    for event in model.events:
        amplitude = Fraction(point[event.input_name("amplitude_scale")])
        duration = Fraction(point[event.input_name("duration_scale")])
        start = Fraction(event.scheduled_start_seconds) + Fraction(
            point[event.input_name("timing_jitter_seconds")]
        )
        if amplitude <= 0 or duration <= 0:
            raise ValueError("startup amplitude and duration scales must be positive")
        knots = tuple(
            start + duration * Fraction(t) for t in event.profile.times_seconds
        )
        if start < 0 or knots[-1] > horizon:
            raise ValueError(
                "every startup must begin and finish within the declared horizon"
            )
        transformed.append((event, start, amplitude, duration, knots))
    return transformed


def _response(model, transforms):
    times = sorted(
        {
            Fraction(0),
            Fraction(model.horizon_seconds),
            *(t for *_, knots in transforms for t in knots),
        }
    )

    def values(time, left=False):
        return tuple(
            amplitude * event.profile._at((time - start) / duration, left=left)
            for event, start, amplitude, duration, _ in transforms
        )

    background = Fraction(model.background_load)
    samples = []
    for time in times:
        for side in ("right", "left"):
            loads = values(time, side == "left")
            samples.append((time, side, loads, background + sum(loads)))
    peak = max(samples, key=lambda row: row[3])
    integral = sum(
        (b - a) * (2 * background + sum(values(a)) + sum(values(b, True))) / 2
        for a, b in zip(times, times[1:])
    )
    details = {
        "time_coverage": "all_segments_of_declared_piecewise_linear_profiles",
        "horizon_seconds": model.horizon_seconds,
        "sampling_grid_used": False,
        "continuous_uncertainty_coverage": False,
        "events": [
            {
                "name": event.name,
                "actual_start_seconds": _float(start),
                "startup_end_seconds": _float(knots[-1]),
                "actual_start_exact_seconds": str(start),
                "startup_end_exact_seconds": str(knots[-1]),
                "amplitude_scale": _float(amplitude),
                "duration_scale": _float(duration),
            }
            for event, start, amplitude, duration, knots in transforms
        ],
        "breakpoints": [
            {
                "time_seconds": _float(t),
                "exact_time_seconds": str(t),
                "side": side,
                "event_loads": {
                    e.name: _float(v) for (e, *_), v in zip(transforms, loads)
                },
                "total_load": _float(total),
            }
            for t, side, loads, total in samples
        ],
        "peak_exact": str(peak[3]),
        "peak_time_exact_seconds": str(peak[0]),
        "peak_side": peak[1],
        "integral_exact": str(integral),
    }
    return peak[3], peak[0], integral, details


def _evaluate(claim, realization):
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
                    f"Profile-parameter membership is {check.status}.",
                ),
            ),
        )
    try:
        transforms = _transforms(model, point)
    except ValueError as exc:
        return EvaluationResult(
            contract,
            request,
            "invalid",
            EvaluationPayload("not_evaluated", membership),
            tuple(evidence),
            (Diagnostic("invalid_startup_realization", "profile", str(exc)),),
        )
    try:
        peak, when, integral, details = _response(model, transforms)
        numbers = {
            **point,
            **{e.control_name: e.scheduled_start_seconds for e in model.events},
            "peak_load": _float(peak),
            "peak_time": _float(when),
            "integrated_load": _float(integral),
            "trajectory_residual": 0,
        }
        residuals = {
            "shared_capacity": (
                "capacity_residual",
                _upper(peak - Fraction(model.capacity)),
                model.tolerance,
            )
        }
        if model.integral_limit is not None:
            residuals["startup_exposure"] = (
                "integral_residual",
                _upper(integral - Fraction(model.integral_limit)),
                model.integral_tolerance,
            )
    except (ArithmeticError, ValueError) as exc:
        return EvaluationResult(
            contract,
            request,
            "unresolved",
            EvaluationPayload("unknown", membership),
            tuple(evidence),
            (Diagnostic("startup_arithmetic_unresolved", "response", str(exc)),),
        )
    evidence.append(
        Evidence(
            "profile",
            "fixed_response",
            "exact_piecewise_linear_breakpoints_and_integrals",
            "verified",
            assumptions=(
                "supplied profiles define the full waveform by linear interpolation",
                "fixed schedule plus declared timing jitter",
                "no network or motor dynamics inferred",
            ),
            details=details,
        )
    )
    checks = [
        ConstraintCheck(
            "trajectory_definition",
            "satisfied",
            QuantityValue("trajectory_residual", 0),
            ("profile",),
        )
    ]
    for name, (residual_name, residual, tolerance) in residuals.items():
        if name in claim.requirements:
            numbers[residual_name] = residual
            checks.append(
                ConstraintCheck(
                    name,
                    "satisfied" if residual <= tolerance else "violated",
                    QuantityValue(residual_name, residual),
                    ("profile",),
                )
            )
    return EvaluationResult(
        contract,
        request,
        "completed",
        EvaluationPayload(
            "feasible",
            membership,
            tuple(QuantityValue(n, v) for n, v in numbers.items()),
            tuple(checks),
            included_constraints=("trajectory_definition",),
        ),
        tuple(evidence),
    )


def _audit(claim):
    evaluations, scenarios, unresolved, evidence = {}, [], [], []
    for scenario in claim.domain.scenarios:
        result = _evaluate(claim, scenario.values)
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
            profile = next(e for e in result.evidence if e.evidence_id == "profile")
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
                        "fixed_schedule_failure",
                        "recourse_infeasibility",
                        "declared_fixed_schedule_violation",
                        "verified",
                        details={
                            "evaluation_ref": result.ref.to_dict(),
                            "scope": "fixed schedule only; alternative schedules not ruled out",
                        },
                    ),
                )
            )
            witness = Witness(
                "fixed_policy_failure",
                (result.request["realization"],),
                affected,
                ("witness_membership",),
                ("fixed_schedule_failure",),
                details={
                    "evaluation_ref": result.ref.to_dict(),
                    "peak_time_seconds": next(
                        v.value
                        for v in result.payload.values
                        if v.quantity_ref == "peak_time"
                    ),
                    "profile_evidence": profile.details,
                },
            )
            break
    if complete:
        evidence.append(
            Evidence(
                "coverage",
                "domain_coverage",
                "finite_scenarios_and_all_profile_segments",
                "verified",
                measurements=(Measurement("scenario_count", len(scenarios), "count"),),
                details={
                    "horizon_seconds": claim.adapter.horizon_seconds,
                    "continuous_time_for_declared_interpolation": True,
                    "continuous_uncertainty_coverage": False,
                },
            )
        )
    coverage = Coverage(
        "complete_finite" if complete else "partial",
        evaluated_support={
            "scenarios": scenarios,
            "time_coverage": "all_segments_of_declared_piecewise_linear_profiles",
            "horizon_seconds": claim.adapter.horizon_seconds,
        },
        unexplored_support={"unresolved_scenarios": unresolved},
        evidence_refs=("coverage",) if complete else (),
    )
    verdict = "fail" if witness else "pass" if complete else "inconclusive"
    return RobustnessResult(
        claim.contract,
        {"query": "audit"},
        "completed" if complete else "unresolved",
        RobustnessPayload(
            verdict,
            coverage,
            tuple(r.ref for r in evaluations.values()),
            witness=witness,
        ),
        tuple(evidence),
        tuple(
            Diagnostic(
                "scenario_unresolved",
                name,
                "A declared scenario could not be evaluated; complete coverage is unavailable.",
            )
            for name in unresolved
        ),
        tuple(evaluations.values()),
    )
