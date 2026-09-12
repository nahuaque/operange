"""Continuous controller enclosures, true command rounding, and threshold bounds."""

from dataclasses import replace
from fractions import Fraction
import json

import pytest

from operange import (
    AffineOutput,
    AffineTerm,
    BoxSet,
    BudgetSet,
    ChangeComparison,
    EllipsoidSet,
    FrozenController,
    Intersection,
    LinearConstraint,
    Parameter,
    PolytopeSet,
    result_from_json,
)
from operange import linear
from operange.results import SolveEvidence
from examples.frozen_controllers import envelope_example, run_envelope_example
from tests.test_frozen_controllers import portable, single_rule
from tests.test_linear_process import small_model


def evidence(result, name):
    return next(e for e in result.evidence if e.evidence_id == name)


def test_full_boiler_envelope_fails_and_same_controller_is_restored():
    result = run_envelope_example()
    audit = result_from_json(json.dumps(result["audit"]))
    assert audit.payload.verdict == "fail"  # nosec B101
    assert audit.payload.coverage.method == "analytical_domain"  # nosec B101
    assert audit.payload.witness.kind == "fixed_policy_failure"  # nosec B101
    claim = envelope_example()
    witness = claim.evaluate_result(audit.payload.witness.realizations[0])
    assert witness.payload.feasibility == "infeasible"  # nosec B101
    comparison = ChangeComparison.from_json(json.dumps(result["comparison"]))
    assert comparison.candidates[0].transition == "restored"  # nosec B101
    frozen = FrozenController.from_json(json.dumps(result["frozen_controller"]))
    replay = result_from_json(json.dumps(result["replayed_audit"]))
    assert frozen.audit_result().result_id == replay.result_id  # nosec B101
    assert replay.payload.verdict == "pass"  # nosec B101
    assert replay.contract.domain == audit.contract.domain  # nosec B101
    assert replay.contract.operating == audit.contract.operating  # nosec B101
    portable(audit)
    portable(replay)


@pytest.mark.parametrize("kind", ("polytope", "ellipsoid", "budget"))
def test_coupled_domains_pass_using_support_bounds(kind):
    claim = envelope_example()
    box = claim.domain
    domain = (
        PolytopeSet(
            box,
            (LinearConstraint("combined_deviation", {"dryer": 1, "evaporator": 1}, 1),),
        )
        if kind == "polytope"
        else BudgetSet(box, 1)
        if kind == "budget"
        else EllipsoidSet(
            box.space,
            {
                "dryer": {"dryer": 1, "evaporator": 0},
                "evaporator": {"dryer": 0, "evaporator": 1},
            },
        )
    )
    result = replace(claim, domain=domain).audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    assert result.payload.coverage.method == "analytical_domain"  # nosec B101
    assert evidence(result, "bound:shared_fuel").measurements[0].value < 30  # nosec B101
    portable(result)


def test_identity_rule_proves_zero_tolerance_balance_and_exact_control_bounds():
    box = BoxSet((Parameter("load", "MW", 0.5, 0, 1, 0.5, "Test"),))
    claim = small_model().as_claim(box, controller=single_rule())
    result = claim.audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    assert evidence(result, "bound:service").measurements[0].value == 0  # nosec B101
    assert evidence(result, "bound:control_upper:power").measurements[0].value == 0  # nosec B101


def test_rounding_can_break_the_interior_when_ideal_affine_map_and_endpoints_pass():
    box = BoxSet((Parameter("load", "MW", 0, 0, 2, 1, "Rounding example"),))
    model = small_model(bounds=(0, 2**54))
    model = replace(model, outputs=(replace(model.outputs[0], offset=-(2**53)),))
    claim = model.as_claim(box, controller=single_rule(offset=2**53))
    assert all(
        claim.evaluate_result({"load": x}).payload.feasibility == "feasible"
        for x in (0, 2)
    )  # nosec B101
    assert claim.evaluate_result({"load": 1}).payload.feasibility == "infeasible"  # nosec B101
    result = claim.audit_result()
    assert result.payload.verdict == "inconclusive"  # nosec B101
    assert (
        evidence(result, "bound:service").details["command_rounding_error_exact"] == "1"
    )  # nosec B101
    portable(result)


def test_unselected_service_does_not_hide_equipment_failure():
    claim = envelope_example()
    rule = replace(claim.adapter.controller.rules[0], offset=100)
    controller = replace(
        claim.adapter.controller, rules=(rule, claim.adapter.controller.rules[1])
    )
    result = claim.with_controller(controller).audit_result()
    assert result.payload.verdict == "fail"  # nosec B101
    assert "control_upper:boiler_a" in result.payload.witness.affected_constraints  # nosec B101


def test_unsupported_membership_only_domain_and_unresolved_support(monkeypatch):
    claim = envelope_example()
    assert (
        replace(claim, domain=Intersection((claim.domain, claim.domain)))
        .audit_result()
        .execution
        == "unsupported"
    )  # nosec B101
    polytope = PolytopeSet(
        claim.domain, (LinearConstraint("joint", {"dryer": 1}, 0.5),)
    )

    def unresolved(*args, **kwargs):
        raise ValueError("Injected unresolved support")

    monkeypatch.setattr(PolytopeSet, "maximize_linear", unresolved)
    result = replace(claim, domain=polytope).audit_result()
    assert result.payload.verdict == "inconclusive"  # nosec B101
    assert result.payload.coverage.method == "partial"  # nosec B101
    portable(result)


def test_undeclared_output_overflow_cannot_establish_a_continuous_pass():
    box = BoxSet((Parameter("load", "MW", 0, 0, 2, 1, "Test"),))
    model = small_model(bounds=(0, 2))
    extra = AffineOutput(
        "large",
        "MW",
        "thermal_power",
        0,
        (AffineTerm("load", 1e308, "MW/MW"),),
        "Overflow test",
    )
    claim = replace(model, outputs=model.outputs + (extra,)).as_claim(
        box, controller=single_rule()
    )
    assert claim.audit_result().payload.verdict == "inconclusive"  # nosec B101


def test_unexportable_residual_cannot_establish_a_continuous_pass():
    box = BoxSet((Parameter("load", "MW", 0, -1e308, 0, 1e308, "Test"),))
    model = small_model(bounds=(-1e308, 1e308))
    model = replace(
        model,
        outputs=(replace(model.outputs[0], terms=(AffineTerm("load", 1, "MW/MW"),)),),
        requirements=(replace(model.requirements[0], limit=1e308, relation="le"),),
    )
    claim = model.as_claim(box, controller=single_rule(coefficient=0))
    assert claim.evaluate_result({"load": 0}).payload.feasibility == "feasible"  # nosec B101
    assert claim.evaluate_result({"load": -1e308}).execution == "unresolved"  # nosec B101
    assert claim.audit_result().payload.verdict == "inconclusive"  # nosec B101


def test_controller_distance_checks_actual_commands_and_closes_bounds():
    claim = envelope_example()
    result = claim.breaking_result(
        constraints=("shared_fuel",), violation_margins={"shared_fuel": 0.01}
    )
    search = result.payload.search
    assert search.resolution == "minimum_verified"  # nosec B101
    expected = (30.01 / 1.55 - 16) / 4
    assert search.lower.value == pytest.approx(expected)  # nosec B101
    assert search.upper.value == pytest.approx(expected)  # nosec B101
    assert search.upper.value - search.lower.value <= search.tolerance  # nosec B101
    replay = claim.evaluate_result(result.payload.witness.realizations[0])
    commands = claim.adapter.controller.commands(replay.request["realization"])
    fuel = Fraction(1.25) * Fraction(commands["boiler_a"]) + 2 * Fraction(
        commands["boiler_b"]
    )
    assert fuel - 30 >= Fraction(0.01)  # nosec B101
    assert evidence(result, "controller_enclosure").outcome == "verified"  # nosec B101
    portable(result)
    boundary = claim.boundary_result(constraints=("shared_fuel",))
    assert boundary.payload.search.resolution == "minimum_verified"  # nosec B101
    assert boundary.payload.verdict == "not_assessed"  # nosec B101
    portable(boundary)


def test_controller_polytope_distance_unreachable_and_invalid_targets():
    claim = envelope_example()
    domain = PolytopeSet(
        claim.domain, (LinearConstraint("joint", {"dryer": 1, "evaporator": 1}, 1),)
    )
    result = replace(claim, domain=domain).breaking_result(
        constraints=("shared_fuel",), violation_margins={"shared_fuel": 0.01}
    )
    assert result.payload.search.resolution == "unreachable"  # nosec B101
    assert result.payload.witness is None  # nosec B101
    portable(result)
    for constraints in (
        (),
        ("unknown",),
        ("shared_fuel", "shared_fuel"),
        "shared_fuel",
    ):
        assert claim.boundary_result(constraints=constraints).execution == "invalid"  # nosec B101
    assert (
        claim.breaking_result(
            constraints=("shared_fuel",), violation_margins={"shared_fuel": 0}
        ).execution
        == "invalid"
    )  # nosec B101


def test_solver_failure_never_certifies_minimum(monkeypatch):
    monkeypatch.setattr(
        linear,
        "solve_lp",
        lambda *a, **k: (None, SolveEvidence("infeasible", "Injected", "test", 1e-9)),
    )
    result = envelope_example().breaking_result(
        constraints=("shared_fuel",), violation_margins={"shared_fuel": 0.01}
    )
    assert result.payload.search.resolution == "unresolved"  # nosec B101
    assert result.payload.search.lower.value == 0  # nosec B101
    portable(result)
