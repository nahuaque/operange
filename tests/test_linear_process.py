"""Linear recourse acceptance, scope boundaries and adversarial solver results."""

from dataclasses import replace
from fractions import Fraction
import json
from types import SimpleNamespace

import pytest

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    Coordinate,
    DecisionRule,
    FiniteSet,
    LinearControl,
    LinearProcessAdapter,
    Parameter,
    ParameterSpace,
    RecoursePolicy,
    Scenario,
    result_from_json,
)
from operange import linear
from operange.results import SolveEvidence
from examples.linear_dispatch import example, run_example


def small_model(*, tolerance=0, scale=1, limits=(), bounds=(0, 1)):
    space = ParameterSpace((Coordinate("load", "MW", "Test load"),))
    return LinearProcessAdapter(
        "Small dispatch",
        space,
        (
            AffineOutput(
                "margin",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("power", 1, "MW/MW"), AffineTerm("load", -1, "MW/MW")),
                "Power minus demand",
            ),
        ),
        (AffineRequirement("service", "margin", 0, "ge", tolerance, scale),),
        (LinearControl("power", "MW", *bounds),),
        limits,
    )


def cases(model, *loads):
    return FiniteSet(
        model.input_space,
        tuple(
            Scenario(str(i), {"load": load}, "Test case")
            for i, load in enumerate(loads)
        ),
    )


def portable(result):
    for compact in (False, True):
        restored = result_from_json(result.to_json(compact=compact))
        assert restored.to_dict() == result.to_dict()  # nosec B101


def test_shared_boilers_fixed_adjustable_and_capacity_change():
    exported = run_example()
    results = {n: result_from_json(json.dumps(v)) for n, v in exported.items()}
    assert results["nominal"].payload.feasibility == "feasible"  # nosec B101
    assert results["fixed"].payload.verdict == "fail"  # nosec B101
    adjustable = results["adjustable"]
    assert adjustable.payload.verdict == "fail"  # nosec B101
    assert adjustable.payload.coverage.method == "complete_finite"  # nosec B101
    assert adjustable.payload.witness.realizations == ({"dryer": 12, "evaporator": 8},)  # nosec B101
    by_point = {
        tuple(e.request["realization"].values()): e
        for e in adjustable.supporting_evaluations
    }
    assert sum(e.payload.feasibility == "feasible" for e in by_point.values()) == 3  # nosec B101
    upgraded = results["enlarged_fuel_supply"]
    assert upgraded.payload.verdict == "pass"  # nosec B101
    assert upgraded.contract.domain == adjustable.contract.domain  # nosec B101
    assert upgraded.contract.operating == adjustable.contract.operating  # nosec B101
    for evaluation in upgraded.supporting_evaluations:
        values = {v.quantity_ref: v.value for v in evaluation.payload.values}
        a, b = Fraction(values["boiler_a"]), Fraction(values["boiler_b"])
        demand = Fraction(values["dryer"]) + Fraction(values["evaporator"])
        assert 0 <= a <= 12 and 0 <= b <= 10  # nosec B101
        assert abs(a + b - demand) <= Fraction(1e-8)  # nosec B101
        assert Fraction(1.25) * a + 2 * b <= 32 + Fraction(1e-8)  # nosec B101
        assert evaluation.payload.objective is None  # nosec B101
    for result in results.values():
        portable(result)


def test_joint_certificate_is_independently_checkable_and_not_individual_violations():
    model, domain = example()
    result = model.as_claim(domain).evaluate_result({"dryer": 12, "evaporator": 8})
    assert result.payload.feasibility == "infeasible"  # nosec B101
    proof = next(e for e in result.evidence if e.subject == "recourse_infeasibility")
    data = proof.details
    rows = [[Fraction(a) for a in row] for row in data["rows_exact"]]
    rhs = [Fraction(b) for b in data["upper_exact"]]
    multipliers = [Fraction(v) for v in data["multipliers"]]
    assert all(v >= 0 for v in multipliers)  # nosec B101
    assert sum(v > 0 for v in multipliers) >= 2  # nosec B101
    weighted = [
        sum(v * a for v, a in zip(multipliers, column)) for column in zip(*rows)
    ]
    gap = sum(min(0, a) for a in weighted) - sum(
        v * b for v, b in zip(multipliers, rhs)
    )
    assert gap > 0 and gap == Fraction(data["contradiction_gap_exact"])  # nosec B101
    assert all(c.assessment == "unknown" for c in result.payload.constraint_checks)  # nosec B101
    assert not any(v.quantity_ref == "boiler_a" for v in result.payload.values)  # nosec B101
    # Independently: 20 MW needs at least 8 MW from B after A reaches 12 MW.
    assert Fraction(1.25) * 12 + 2 * 8 == 31 > 30  # nosec B101
    portable(result)


def test_operating_limits_always_apply_when_selecting_requirements():
    model = small_model()
    model = replace(
        model,
        requirements=model.requirements
        + (AffineRequirement("reserve", "margin", 0.9, "ge", 0),),
        operating_limits=(AffineRequirement("equipment", "margin", 0.25, tolerance=0),),
    )
    domain = cases(model, 0.5)
    assert model.as_claim(domain).audit_result().payload.verdict == "fail"  # nosec B101
    selected = model.as_claim(domain, requirements=("service",))
    result = selected.evaluate_result({"load": 0.5})
    assert result.payload.feasibility == "feasible"  # nosec B101
    assert "equipment" in result.payload.included_constraints  # nosec B101
    assert "reserve" not in result.payload.included_constraints  # nosec B101
    with pytest.raises(ValueError, match="undeclared engineering requirement"):
        model.as_claim(domain, requirements=("equipment",))
    impossible = replace(
        model,
        operating_limits=(AffineRequirement("equipment", "margin", -0.1, tolerance=0),),
    )
    assert (
        impossible.as_claim(domain, requirements=("service",))
        .audit_result()
        .payload.verdict
        == "fail"
    )  # nosec B101


@pytest.mark.parametrize(
    "fixed,expected", [(0.75, "pass"), (0.25, "fail"), (-0.1, "fail"), (1.1, "fail")]
)
def test_fixed_controls_remain_subject_to_service_and_physical_bounds(fixed, expected):
    model = small_model()
    policy = RecoursePolicy(
        "fixed", (DecisionRule("power", "MW", "operation", fixed_value=fixed),)
    )
    result = model.as_claim(cases(model, 0.5), recourse=policy).audit_result()
    assert result.payload.verdict == expected  # nosec B101
    if expected == "fail":
        assert result.payload.witness.kind == "fixed_policy_failure"  # nosec B101
    portable(result)


def test_mixed_fixed_and_adjustable_controls_and_zero_width_bounds():
    model, domain = example()
    policy = RecoursePolicy(
        "static",
        (
            DecisionRule("boiler_a", "MW", "operation", fixed_value=12),
            DecisionRule("boiler_b", "MW", "operation", model.input_space.names),
        ),
    )
    result = model.as_claim(domain, recourse=policy).evaluate_result(
        {"dryer": 10, "evaporator": 6}
    )
    values = {v.quantity_ref: v.value for v in result.payload.values}
    assert result.payload.feasibility == "feasible"  # nosec B101
    assert values["boiler_a"] == 12 and abs(values["boiler_b"] - 4) <= 1e-8  # nosec B101
    constant = small_model(bounds=(0.5, 0.5))
    assert (
        constant.as_claim(cases(constant, 0.5)).audit_result().payload.verdict == "pass"
    )  # nosec B101
    assert (
        constant.as_claim(cases(constant, 0.6)).audit_result().payload.verdict == "fail"
    )  # nosec B101


@pytest.mark.parametrize("status", [1, 2, 3, 4])
def test_failed_solver_does_not_imply_physical_infeasibility(monkeypatch, status):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False,
            status=status,
            message="Injected solver failure",
            x=None,
        ),
    )
    model = small_model()
    result = model.as_claim(cases(model, 0.5)).audit_result()
    assert result.execution == "unresolved"  # nosec B101
    assert result.payload.verdict == "inconclusive" and result.payload.witness is None  # nosec B101
    assert result.payload.coverage.method == "partial"  # nosec B101
    portable(result)


def test_exact_checks_reject_a_candidate_within_solver_but_outside_physical_tolerance(
    monkeypatch,
):
    # Scaling hides the 0.5 MW violation inside a 1e-9 numerical LP tolerance.
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=True,
            status=0,
            message="Injected approximate success",
            x=[0.0, 0.0],
        ),
    )
    model = small_model(scale=1e12)
    result = model.as_claim(cases(model, 0.5)).evaluate_result({"load": 0.5})
    assert result.execution == "unresolved" and result.payload.feasibility == "unknown"  # nosec B101
    assert not any(e.subject == "recourse_infeasibility" for e in result.evidence)  # nosec B101
    portable(result)


def test_bogus_dual_multipliers_do_not_certify_failure(monkeypatch):
    model = small_model(
        limits=(AffineRequirement("upper", "margin", -0.25, tolerance=0),)
    )
    calls = iter(
        [
            (None, SolveEvidence("unresolved", "Injected", "test", 1e-9)),
            ([1.0, 0.0, 0.0], SolveEvidence("optimal", "Injected", "test", 1e-9)),
        ]
    )
    monkeypatch.setattr(linear, "solve_lp", lambda *a, **k: next(calls))
    result = model.as_claim(cases(model, 0.5)).audit_result()
    assert result.payload.verdict == "inconclusive"  # nosec B101
    assert result.payload.witness is None  # nosec B101


def test_verified_failure_survives_other_unresolved_scenarios(monkeypatch):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False,
            status=1,
            message="Time limit",
            x=None,
        ),
    )
    model = small_model()
    result = model.as_claim(cases(model, 0.5, 2)).audit_result()
    assert result.execution == "unresolved" and result.payload.verdict == "fail"  # nosec B101
    assert result.payload.witness.realizations == ({"load": 2},)  # nosec B101
    assert result.payload.coverage.unexplored_support["unresolved_scenarios"] == ("0",)  # nosec B101
    portable(result)


def test_duplicate_scenarios_retain_identities_and_use_physical_not_normalized_inputs():
    model = small_model()
    model = replace(
        model,
        input_space=ParameterSpace((Coordinate("load", "MW", "Test", 1e16, 1e16),)),
    )
    result = model.as_claim(cases(model, 0, 1, 1)).audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    scenarios = result.payload.coverage.evaluated_support["scenarios"]
    assert len(scenarios) == 3 and len(result.supporting_evaluations) == 2  # nosec B101
    assert {
        e.request["realization"]["load"] for e in result.supporting_evaluations
    } == {0, 1}  # nosec B101
    portable(result)


@pytest.mark.parametrize(
    "mode,stage,observes",
    [
        ("causal", "operation", ("load",)),
        ("perfect_foresight", "operation", ("load",)),
        ("static", "preparation", ("load",)),
        ("static", "operation", ()),
        ("static", "operation", ("future",)),
    ],
)
def test_unsupported_information_permissions_are_never_silently_relaxed(
    mode, stage, observes
):
    model = small_model()
    policy = RecoursePolicy(mode, (DecisionRule("power", "MW", stage, observes),))
    result = model.as_claim(cases(model, 0.5), recourse=policy).audit_result()
    assert (
        result.execution == "unsupported" and result.payload.verdict == "not_assessed"
    )  # nosec B101
    portable(result)


def test_missing_control_permission_is_unsupported():
    model = small_model()
    result = model.as_claim(
        cases(model, 0), recourse=RecoursePolicy("fixed", ())
    ).audit_result()
    assert result.execution == "unsupported"  # nosec B101


def test_continuous_domain_point_evaluation_does_not_grant_a_recourse_audit_or_derivative():
    model = small_model()
    domain = BoxSet((Parameter("load", "MW", 0.5, 0, 1, 0.5, "Test box"),))
    claim = model.as_claim(domain)
    assert claim.evaluate_result({"load": 0.5}).payload.feasibility == "feasible"  # nosec B101
    for result in (
        claim.audit_result(),
        claim.sensitivity_result({"load": 0.5}),
        claim.boundary_result(),
        claim.breaking_result(),
    ):
        assert result.execution == "unsupported"  # nosec B101
    assert claim.evaluate_result({"load": 2}).execution == "invalid"  # nosec B101
    assert claim.evaluate_result({"unknown": 0}).execution == "invalid"  # nosec B101
    assert claim.evaluate_result().execution == "invalid"  # nosec B101
    assert claim.evaluate_result({"load": 0.5}, optimize=True).execution == "invalid"  # nosec B101


@pytest.mark.parametrize("load,expected", [(1e-8, "pass"), (1.1e-8, "fail")])
def test_declared_requirement_tolerance_is_part_of_both_proofs(load, expected):
    model = small_model(tolerance=1e-8, bounds=(0, 0))
    result = model.as_claim(cases(model, load)).audit_result()
    assert result.payload.verdict == expected  # nosec B101


def test_negative_and_large_control_origins_are_supported():
    for bounds, loads in (((-4, -2), (-3, -2)), ((1e16, 1e16 + 2), (1e16, 1e16 + 2))):
        model = small_model(bounds=bounds)
        result = model.as_claim(cases(model, *loads)).audit_result()
        assert result.payload.verdict == "pass"  # nosec B101


def test_unexportable_output_is_unresolved_not_a_pass():
    model = small_model()
    model = replace(
        model,
        outputs=model.outputs
        + (
            AffineOutput(
                "overflow",
                "MW",
                "thermal_power",
                1e308,
                (AffineTerm("power", 1e308, "MW/MW"),),
                "Extreme output",
            ),
        ),
    )
    result = model.as_claim(cases(model, 1)).audit_result()
    assert result.payload.verdict == "inconclusive"  # nosec B101
    portable(result)


@pytest.mark.parametrize(
    "bounds", [(1, 0), (0, float("inf")), (float("nan"), 1), (False, 1)]
)
def test_control_bound_validation(bounds):
    with pytest.raises(ValueError):
        LinearControl("x", "MW", *bounds)


def test_declarations_validate_units_names_and_round_trip():
    model = small_model()
    assert LinearProcessAdapter(**model.to_dict()) == model  # nosec B101
    with pytest.raises(ValueError, match="units"):
        replace(model, controls=(LinearControl("power", "kW", 0, 1),))
    with pytest.raises(ValueError):
        replace(model, controls=model.controls * 2)
    with pytest.raises(ValueError):
        replace(model, operating_limits=model.requirements)
    with pytest.raises(ValueError):
        replace(
            model, requirements=(AffineRequirement("control_upper:power", "margin", 1),)
        )
    with pytest.raises(ValueError):
        replace(model, solver_tolerance=0)
    with pytest.raises(ValueError):
        replace(model, controls=())
