"""Whole-box policy search: independent finite oracles and large-domain replay."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import json

import pytest

from operange import (
    AffineControlRule,
    AffineController,
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    ControlTarget,
    ControlTrackingObjective,
    FiniteSet,
    FrozenController,
    LinearControl,
    LinearObjective,
    LinearProcessAdapter,
    Parameter,
    Scenario,
    verify_result,
)
from operange import _controller_synthesis as synthesis
from examples.scalable_controller import example, run_example


@pytest.fixture
def cp():
    return pytest.importorskip("cvxpy")


def objective(result):
    return next(e.details for e in result.evidence if e.evidence_id == "objective")


def test_twenty_loads_need_no_generators_for_search_audit_or_verification(
    cp, monkeypatch
):
    from operange._certificate_math import Context

    def no_enumeration(*args, **kwargs):
        raise AssertionError("a million-corner box must not enumerate generators")

    monkeypatch.setattr(synthesis, "_generators", no_enumeration)
    monkeypatch.setattr(synthesis, "product", no_enumeration)
    monkeypatch.setattr(Context, "generators", no_enumeration)
    claim = example()
    original = claim.contract
    result = claim.synthesize_controller(max_vertices=1)
    assert result.verified  # nosec B101
    assert result.source.contract == original  # nosec B101
    assert result.request["formulation"] == "box_support_lp"  # nosec B101
    coverage = result.evidence[0]
    assert coverage.method == "box_support_affine_policy_search"  # nosec B101
    assert coverage.details["generator_count"] == 0  # nosec B101
    assert coverage.details["generators"] == ()  # nosec B101
    assert coverage.details["box_support"]["varying_input_count"] == 20  # nosec B101
    bound = objective(result)
    assert bound["guaranteed_value"] == pytest.approx(31, abs=1e-7)  # nosec B101
    assert bound["upper"] - bound["lower"] < 1e-7  # nosec B101
    assert len(bound["attainment_realizations"]) == 1  # nosec B101
    assert bound["generator_minimization_values_exact"] == ()  # nosec B101
    assert bound["optimality"] == "not_certified"  # nosec B101
    assert len(result.audit.supporting_evaluations) < 100  # nosec B101

    def no_solver(*args, **kwargs):
        raise AssertionError("replay and certificate verification must not solve")

    monkeypatch.setattr(cp.Problem, "solve", no_solver)
    monkeypatch.setattr(synthesis, "load_cvxpy", no_solver)
    frozen = FrozenController.from_json(result.freeze().to_json())
    assert frozen.audit_result() == result.audit  # nosec B101
    for compact in (False, True):
        report = verify_result(
            result.audit.to_json(compact=compact),
            expected_contract_id=result.audit.contract_ref.artifact_id,
        )
        assert report.verified, report.to_json()  # nosec B101
        assert json.loads(result.to_json(compact=compact))["verified"]  # nosec B101
    for i in range(7):
        point = {
            n: 0.75 + ((j + i) % 7) / 24 for j, n in enumerate(claim.domain.space.names)
        }
        replay = frozen.evaluate_result(point)
        assert replay.payload.feasibility == "feasible"  # nosec B101
        assert verify_result(replay).verified  # nosec B101
        commands = result.controller.commands(point)
        fuel = F(1.25) * F(commands["boiler_a"]) + 2 * F(commands["boiler_b"])
        assert fuel <= F(bound["upper_exact"])  # nosec B101


def mixed_unit_claim(sense):
    # Non-centered physical ranges, a constant coordinate, negative slopes,
    # mixed units, and a direct input contribution to the objective.
    box = BoxSet(
        (
            Parameter("y", "MW", 5, 4, 7, 1, "Test"),
            Parameter("bias", "MW", 4, 4, 4, 1, "Test"),
            Parameter("x", "kW", 0, -3000, 2000, 1000, "Test"),
        )
    )
    model = LinearProcessAdapter(
        "Mixed-unit balance",
        box.space,
        (
            AffineOutput(
                "balance",
                "MW",
                "thermal_power",
                -2,
                (
                    AffineTerm("power", 1, "MW/MW"),
                    AffineTerm("x", 0.00025, "MW/kW"),
                    AffineTerm("y", -0.5, "MW/MW"),
                    AffineTerm("bias", 0.25, "MW/MW"),
                ),
                "Delivery minus demand",
            ),
            AffineOutput(
                "cost",
                "MW",
                "thermal_power",
                17,
                (AffineTerm("power", 2, "MW/MW"), AffineTerm("x", -0.001, "MW/kW")),
                "Declared affine performance output",
            ),
        ),
        (AffineRequirement("service", "balance", 0, "ge", 1e-8, 100),),
        (LinearControl("power", "MW", 0, 8),),
        (AffineRequirement("no_excess", "balance", 0, "le", 1e-8, 100),),
        objective=LinearObjective("cost", sense),
    )
    return model.as_claim(box)


@pytest.mark.parametrize("sense,expected", (("minimize", 30.5), ("maximize", 20)))
def test_support_search_matches_enumerated_oracle_with_signed_mixed_unit_rows(
    cp, sense, expected
):
    claim = mixed_unit_claim(sense)
    points = tuple(
        Scenario(str(i), {"x": x, "y": y, "bias": 4}, "All physical corners")
        for i, (x, y) in enumerate(product((-3000, 2000), (4, 7)))
    )
    finite = replace(claim, domain=FiniteSet(claim.domain.space, points))
    supported = claim.synthesize_controller(max_vertices=1)
    enumerated = finite.synthesize_controller()
    assert supported.verified and enumerated.verified  # nosec B101
    for result in (supported, enumerated):
        bound = objective(result)
        assert bound["guaranteed_value"] == pytest.approx(expected, abs=1e-6)  # nosec B101
        attained = [
            claim.adapter.output("cost")._exact_value(
                {**s.values, **result.controller.commands(s.values)}
            )
            for s in points
        ]
        worst = max(attained) if sense == "minimize" else min(attained)
        assert F(bound["lower_exact"]) <= worst <= F(bound["upper_exact"])  # nosec B101
    assert verify_result(supported.audit).verified  # nosec B101
    assert "bias" not in supported.controller.rules[0].observes  # nosec B101
    bound = objective(supported)
    for x, y in product((-2500.5, -1, 1245.75), (4.125, 5, 6.875)):
        point = {"x": x, "y": y, "bias": 4}
        assert supported.claim.evaluate_result(point).payload.feasibility == "feasible"  # nosec B101
        value = claim.adapter.output("cost")._exact_value(
            {**point, **supported.controller.commands(point)}
        )
        assert (
            value <= F(bound["upper_exact"])
            if sense == "minimize"
            else value >= F(bound["lower_exact"])
        )  # nosec B101


def test_large_tracking_box_still_requires_all_generators_before_loading_backend(
    monkeypatch,
):
    def no_backend():
        raise AssertionError("unsupported tracking must be rejected before solving")

    monkeypatch.setattr(synthesis, "load_cvxpy", no_backend)
    result = example().synthesize_controller(
        objective=ControlTrackingObjective((ControlTarget("boiler_a", 10, 1, "MW"),))
    )
    assert result.execution == "unsupported"  # nosec B101
    assert "1048576 generators" in result.diagnostics[0].message  # nosec B101


def test_constant_objective_still_has_an_attained_bound_over_a_large_box(cp):
    claim = example()
    model = replace(
        claim.adapter,
        outputs=claim.adapter.outputs
        + (AffineOutput("constant", "MW", "thermal_power", -7, (), "Test"),),
        objective=LinearObjective("constant"),
    )
    result = replace(claim, adapter=model).synthesize_controller(max_vertices=1)
    assert result.verified  # nosec B101
    assert objective(result)["lower"] == objective(result)["upper"] == -7  # nosec B101


def test_scalable_consumer_exports_independently_verified_audit(cp):
    exported = run_example()
    assert exported["input_count"] == 20  # nosec B101
    assert exported["box_corner_count"] == 1048576  # nosec B101
    assert exported["synthesis"]["verified"]  # nosec B101
    assert exported["verification"]["status"] == "verified"  # nosec B101


def test_objective_bound_covers_interior_rounding_when_ideal_output_is_constant():
    from tests.test_linear_process import small_model

    model = small_model(tolerance=2, bounds=(0, 2**54))
    model = replace(model, outputs=(replace(model.outputs[0], offset=-(2**53)),))
    box = BoxSet((Parameter("load", "MW", 1, 0, 2, 1, "Rounding"),))
    controller = AffineController(
        "Rounding probe",
        (AffineControlRule("power", "MW", 2**53, (AffineTerm("load", 1, "MW/MW"),)),),
        "Known feasible candidate; ideal margin is identically zero",
    )
    claim = model.as_claim(box).with_controller(controller)
    audit = claim.audit_result()
    assert audit.payload.verdict == "pass" and verify_result(audit).verified  # nosec B101
    bound = synthesis._objective_bound(claim, LinearObjective("margin"), None).details
    assert bound["attained_minimization_values_exact"] == ("0",)  # nosec B101
    assert bound["upper"] == 1  # nosec B101
    point = {"load": 1.25}
    actual = model.outputs[0]._exact_value({**point, **controller.commands(point)})
    assert actual == F(3, 4) > F(bound["lower_exact"])  # nosec B101
    assert actual <= F(bound["upper_exact"])  # nosec B101
