"""Adversarial numerical cases checked against independent exact values."""

from dataclasses import replace
from decimal import localcontext
from fractions import Fraction
import json
from types import SimpleNamespace

import pytest

from operange import (
    AffineOutput,
    AffineProcessAdapter,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    Coordinate,
    EllipsoidSet,
    FiniteSet,
    Parameter,
    ParameterSpace,
    PolytopeSet,
    Scenario,
    result_from_json,
)
from operange import linear
from operange.contract_types import digest
from operange.reference import (
    HeatDemand,
    HeatRecoveryClaim,
    HeatRecoveryDesign,
    example_claim,
)


def identity_model(
    space, limit=0.5, unit="MW", coefficient=1, offset=0, tolerance=1e-8
):
    return AffineProcessAdapter(
        "precision regression",
        space,
        (
            AffineOutput(
                "y",
                unit,
                "thermal_power",
                offset,
                (AffineTerm("x", coefficient, f"{unit}/{unit}"),),
                "test",
            ),
        ),
        (AffineRequirement("capacity", "y", limit, tolerance=tolerance),),
    )


@pytest.mark.parametrize(
    "origin,scale", [(0, 1), (1e16, 1e16), (-1e16, 3), (None, None)]
)
@pytest.mark.parametrize("unit,multiplier", [("MW", 1), ("kW", 1000)])
def test_finite_verdict_is_independent_of_normalization_and_unit_scale(
    origin, scale, unit, multiplier
):
    space = ParameterSpace((Coordinate("x", unit, "test", origin, scale),))
    scenarios = (
        Scenario("a", {"x": 0}, "test"),
        Scenario("b", {"x": multiplier}, "test"),
        Scenario("duplicate", {"x": multiplier}, "test"),
    )
    claim = identity_model(space, limit=0.5 * multiplier, unit=unit).as_claim(
        FiniteSet(space, scenarios)
    )
    result = claim.audit_result()
    assert result.payload.verdict == "fail"  # nosec B101
    assert result.payload.coverage.method == "complete_finite"  # nosec B101
    assert len(result.payload.coverage.evaluated_support["scenarios"]) == 3  # nosec B101
    assert len(result.supporting_evaluations) == 2  # nosec B101
    assert result.payload.witness.realizations == ({"x": multiplier},)  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101


def test_finite_support_does_not_tie_distinct_exact_scores():
    space = ParameterSpace((Coordinate("x", "MW", "test", 1e16, 1e16),))
    domain = FiniteSet(
        space, (Scenario("a", {"x": 0}, "test"), Scenario("b", {"x": 1}, "test"))
    )
    result = domain.maximize_linear({"x": 1e16})
    exact = Fraction(1) - Fraction(1e16)
    assert result.point == {"x": 1}  # nosec B101
    assert Fraction(result.lower) <= exact <= Fraction(result.upper)  # nosec B101
    assert result.evidence[-1].details["maximizing_scenarios"] == ("b",)  # nosec B101


@pytest.mark.parametrize("scale", [3, 1e16, 1e-100])
@pytest.mark.parametrize("polytope", [False, True])
def test_continuous_support_cancellation_cannot_hide_a_violation(scale, polytope):
    box = BoxSet((Parameter("x", "MW", 1e16, 1e16, 1e16 + 2, scale, "test"),))
    domain = PolytopeSet(box, ()) if polytope else box
    claim = identity_model(box.space, offset=-1e16).as_claim(domain)
    result = claim.audit_result()
    assert result.payload.verdict != "pass"  # nosec B101
    for evidence in result.evidence:
        if evidence.subject == "requirement_bound" and evidence.outcome == "verified":
            upper = next(
                m.value for m in evidence.measurements if m.name == "residual_upper"
            )
            assert Fraction(upper) >= Fraction(3, 2)  # nosec B101


def test_ellipsoid_upper_bound_rounds_outward_against_exact_squared_support():
    space = ParameterSpace((Coordinate("x", "MW", "test", 0, 1),))
    for shape in (0.1, 0.3, 1.1):
        for weight in (0.1, 1.3, 1e10):
            domain = EllipsoidSet(space, {"x": {"x": shape}}, radius=0.7)
            support = domain.maximize_linear({"x": weight})
            exact_squared = Fraction(0.7) ** 2 * Fraction(shape) * Fraction(weight) ** 2
            assert Fraction(support.upper) ** 2 >= exact_squared  # nosec B101


def test_ellipsoid_bound_is_independent_of_callers_decimal_context():
    space = ParameterSpace((Coordinate("x", "MW", "test", 0, 1),))
    domain = EllipsoidSet(space, {"x": {"x": 0.3}})
    expected = domain.maximize_linear({"x": 1e10}).upper
    with localcontext() as context:
        context.prec = 2
        context.Emax = 2
        assert domain.maximize_linear({"x": 1e10}).upper == expected  # nosec B101


def test_exact_affine_residual_does_not_round_tiny_violation_to_zero():
    space = ParameterSpace((Coordinate("x", "MW", "test"),))
    domain = FiniteSet(space, (Scenario("tiny", {"x": 0.5}, "test"),))
    result = (
        identity_model(space, coefficient=5e-324, limit=0, tolerance=0)
        .as_claim(domain)
        .audit_result()
    )
    assert result.payload.verdict == "fail"  # nosec B101


@pytest.mark.parametrize(
    "demand_delta,verdict", [(-2e-8, "pass"), (7e-9, "inconclusive"), (2e-8, "fail")]
)
def test_solver_error_within_tolerance_preserves_recourse_bounds(
    monkeypatch, demand_delta, verdict
):
    box = BoxSet(
        tuple(
            Parameter(name, "fraction", 0, 0, 0, 1, "test")
            for name in ("source_derating", "demand_increase")
        )
    )
    claim = HeatRecoveryClaim(
        "test", HeatRecoveryDesign(6, 2), box, HeatDemand(8 + demand_delta)
    )
    q = 8 - 5e-9
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            status=0,
            success=True,
            message="feasible slightly suboptimal response",
            x=[q / 4, 3 * q / 4, q],
        ),
    )
    result = claim.audit_result()
    assert result.payload.verdict == verdict  # nosec B101
    objective = result.supporting_evaluations[0].payload.objective
    assert objective.lower.value == q and objective.upper.value == 8  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101
    if verdict == "inconclusive":
        assert result.execution == "unresolved" and result.payload.witness is None  # nosec B101


@pytest.mark.parametrize("quantity", ["requirement_residual", "delivered_heat"])
def test_repeated_quantities_must_agree_on_construction_and_loading(quantity):
    claim = example_claim()
    result = claim.evaluate_result(claim.uncertainty.upper_corner)
    changed_values = tuple(
        replace(v, value=-999) if v.quantity_ref == quantity else v
        for v in result.payload.values
    )
    with pytest.raises(ValueError, match="contradicts stored quantity"):
        replace(result, payload=replace(result.payload, values=changed_values))
    data = result.to_dict()
    for value in data["payload"]["values"]:
        if value["quantity_ref"] == quantity:
            value["value"] = -999
    data["result_id"] = "sha256:" + digest(
        {k: v for k, v in data.items() if k != "result_id"}
    )
    with pytest.raises(ValueError, match="contradicts stored quantity"):
        result_from_json(json.dumps(data))


def test_residuals_shared_between_checks_must_agree_without_a_stored_value():
    claim = example_claim()
    result = claim.evaluate_result(claim.uncertainty.upper_corner)
    check = result.payload.constraint_checks[-1]
    contract = replace(
        result.contract,
        constraints=(
            *result.contract.constraints,
            replace(result.contract.constraints[-1], constraint_id="another_check"),
        ),
    )
    contradictory = replace(
        check,
        constraint_ref="another_check",
        assessment="satisfied",
        residual=replace(check.residual, value=-1),
    )
    payload = replace(
        result.payload,
        values=tuple(
            v
            for v in result.payload.values
            if v.quantity_ref != check.residual.quantity_ref
        ),
        constraint_checks=(*result.payload.constraint_checks, contradictory),
    )
    with pytest.raises(ValueError, match="contradicts stored quantity"):
        replace(result, contract=contract, payload=payload)


@pytest.mark.parametrize("underflow", [False, True])
def test_heat_derivative_arithmetic_failures_are_portable(underflow):
    claim = example_claim()
    if underflow:
        claim = replace(
            claim,
            uncertainty=BoxSet(
                tuple(replace(p, scale=1e-200) for p in claim.uncertainty.parameters)
            ),
        )
    result = claim.sensitivity_result(
        claim.uncertainty.nominal,
        operator="directional",
        side="forward",
        coordinate_space="normalized" if underflow else "physical",
        direction=(1e-200, 1e-200) if underflow else (1e308, 1e308),
    )
    assert result.execution == "unresolved"  # nosec B101
    assert result.payload.derivative is None  # nosec B101
    assert result.diagnostics[0].code == "derivative_arithmetic_unresolved"  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101
