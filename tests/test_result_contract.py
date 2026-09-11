"""Worked engineering cases, failure semantics, and portable artifact integrity."""

from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import pytest

from operange import (
    Bound,
    Derivative,
    EvaluationResult,
    QuantityValue,
    RobustnessResult,
    SensitivityResult,
    result_from_json,
)
from operange.reference import StorageDesign
from operange import linear
from operange.contract_types import digest
from operange.reference import example_claim
from operange.reference import example_storage_claim

POINT = {"source_derating": 0.1, "demand_increase": 0.05}


def values(result):
    return {v.quantity_ref: v.value for v in result.payload.values}


def resign(data):
    data["result_id"] = "sha256:" + digest(
        {k: v for k, v in data.items() if k != "result_id"}
    )
    return json.dumps(data)


def test_evaluation_keeps_physics_separate_from_service():
    c = example_claim()
    result = c.evaluate_result(POINT)
    assert isinstance(result, EvaluationResult)  # nosec B101
    assert result.execution == "completed"  # nosec B101
    assert result.payload.feasibility == "feasible"  # nosec B101
    assert values(result) == pytest.approx(
        {
            **POINT,
            "delivered_heat": 7.2,
            "required_heat": 6.3,
            "delivery_margin": 0.9,
            "requirement_residual": -0.9,
            "electrical_power": 1.8,
            "source_heat": 5.4,
        }
    )  # nosec B101
    assert result.payload.objective.optimality == "verified"  # nosec B101
    worst = c.evaluate_result(c.uncertainty.upper_corner)
    assert worst.payload.feasibility == "feasible"  # nosec B101
    assert "heat_demand" not in worst.payload.included_constraints  # nosec B101
    assert worst.payload.constraint_checks[-1].assessment == "violated"  # nosec B101
    assert values(worst)["delivery_margin"] == pytest.approx(-1.5)  # nosec B101
    assert c.audit_result().payload.verdict == "fail"  # nosec B101


def test_heat_search_boundary_and_strict_witness_are_different():
    c = example_claim()
    boundary, breaking = c.boundary_result(), c.breaking_result()
    assert isinstance(breaking, RobustnessResult)  # nosec B101
    assert boundary.payload.verdict == "not_assessed"  # nosec B101
    assert boundary.payload.witness is None  # nosec B101
    assert boundary.payload.search.candidate_distance == pytest.approx(4 / 7)  # nosec B101
    assert breaking.payload.verdict == "fail"  # nosec B101
    assert breaking.payload.search.resolution == "minimum_verified"  # nosec B101
    assert breaking.payload.search.lower.value == pytest.approx(201 / 350)  # nosec B101
    assert dict(breaking.payload.witness.realizations[0]) == pytest.approx(
        dict.fromkeys(POINT, 201 / 1400)
    )  # nosec B101
    assert values(breaking.supporting_evaluations[0])[
        "delivered_heat"
    ] == pytest.approx(1199 / 175)  # nosec B101


def test_unreachable_threshold_does_not_pass_claim():
    result = example_claim().breaking_result(shortfall_margin_mw=2)
    assert result.execution == "completed"  # nosec B101
    assert result.payload.search.resolution == "unreachable"  # nosec B101
    assert result.payload.verdict == "not_assessed"  # nosec B101
    assert result.payload.witness is None  # nosec B101


def test_verified_witness_survives_unresolved_minimum(monkeypatch):
    original = linear.linprog

    def fail_compressor(objective, **kwargs):
        if objective == [0, 0, 1] and kwargs["A_ub"][-1][0] == 0:
            return SimpleNamespace(
                success=False, status=1, message="time limit", x=None
            )
        return original(objective, **kwargs)

    monkeypatch.setattr(linear, "linprog", fail_compressor)
    result = example_claim().breaking_result()
    assert result.execution == "unresolved"  # nosec B101
    assert result.payload.verdict == "fail"  # nosec B101
    assert result.payload.search.resolution == "bounded"  # nosec B101
    assert result.payload.search.lower.value == 0  # nosec B101
    assert result.payload.search.upper.value == pytest.approx(201 / 350)  # nosec B101
    assert any(e.outcome == "unresolved" for e in result.evidence)  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101


def test_solver_failure_is_unknown_not_physical_infeasibility(monkeypatch):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False, status=1, message="timeout", x=None
        ),
    )
    c = example_claim()
    evaluation = c.evaluate_result(POINT)
    assert evaluation.execution == "unresolved"  # nosec B101
    assert evaluation.payload.feasibility == "unknown"  # nosec B101
    assert "delivered_heat" not in values(evaluation)  # nosec B101
    assert c.audit_result().payload.verdict == "inconclusive"  # nosec B101
    assert c.sensitivity_result(POINT).payload.availability == "unknown"  # nosec B101
    s = example_storage_claim().audit_result()
    assert s.payload.verdict == "inconclusive"  # nosec B101
    assert s.payload.witness is None  # nosec B101
    assert s.supporting_evaluations[0].payload.feasibility == "unknown"  # nosec B101


@pytest.mark.parametrize(
    "point",
    [
        {},
        {**POINT, "source_derating": 0.3},
        {**POINT, "source_derating": float("nan")},
        {**POINT, "demand_increase": True},
        None,
    ],
)
def test_invalid_evaluation_is_portable_and_not_evaluated(point):
    result = example_claim().evaluate_result(point)
    assert result.execution == "invalid"  # nosec B101
    assert result.payload.feasibility == "not_evaluated"  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101


@pytest.mark.parametrize("margin", [0, -1, float("inf"), True])
def test_invalid_search_has_no_verdict(margin):
    result = example_claim().breaking_result(shortfall_margin_mw=margin)
    assert result.execution == "invalid"  # nosec B101
    assert result.payload.verdict == "not_assessed"  # nosec B101


def test_smooth_analytical_jacobian_normalization_and_coordinate_order():
    c = example_claim()
    result = c.sensitivity_result(POINT)
    assert isinstance(result, SensitivityResult)  # nosec B101
    assert result.payload.derivative.values == ((-8, 0), (0, 6), (-8, -6), (8, 6))  # nosec B101
    normalized = c.sensitivity_result(
        POINT,
        coordinate_space="normalized",
        outputs=("delivery_margin",),
        inputs=("demand_increase", "source_derating"),
    )
    derivative = normalized.payload.derivative
    assert derivative.values == ((-1.5, -2),)  # nosec B101
    assert derivative.input_coordinates == ("demand_increase", "source_derating")  # nosec B101
    assert derivative.scales == (0.25, 0.25)  # nosec B101
    assert derivative.units == (("MW/1", "MW/1"),)  # nosec B101
    # A numerical cross-check independent of the exported derivative matrix.
    h = 1e-5
    plus = values(
        c.evaluate_result({**POINT, "source_derating": POINT["source_derating"] + h})
    )["delivery_margin"]
    minus = values(
        c.evaluate_result({**POINT, "source_derating": POINT["source_derating"] - h})
    )["delivery_margin"]
    assert (plus - minus) / (2 * h) == pytest.approx(-8)  # nosec B101


def test_directional_physical_and_normalized_agree():
    c = example_claim()
    physical = c.sensitivity_result(
        POINT, operator="directional", direction=(0.01, 0), outputs=("delivery_margin",)
    )
    normalized = c.sensitivity_result(
        POINT,
        operator="directional",
        coordinate_space="normalized",
        direction=(0.04, 0),
        outputs=("delivery_margin",),
    )
    assert physical.payload.derivative.values == pytest.approx((-0.08,))  # nosec B101
    assert physical.payload.derivative.values == normalized.payload.derivative.values  # nosec B101
    assert physical.payload.derivative.units == (("MW",),)  # nosec B101


def test_kink_one_sided_derivatives_and_unaffected_outputs():
    c = example_claim()
    c = replace(c, design=replace(c.design, power_capacity_mw=1.8))
    both = c.sensitivity_result(POINT)
    assert both.payload.availability == "undefined"  # nosec B101
    assert both.execution == "completed"  # nosec B101
    forward = c.sensitivity_result(POINT, side="forward", outputs=("delivered_heat",))
    backward = c.sensitivity_result(POINT, side="backward", outputs=("delivered_heat",))
    assert forward.payload.derivative.values == ((-8, 0),)  # nosec B101
    assert backward.payload.derivative.values == ((0, 0),)  # nosec B101
    assert (
        c.sensitivity_result(POINT, outputs=("required_heat",)).payload.availability
        == "available"
    )  # nosec B101
    assert (
        c.sensitivity_result(POINT, inputs=("demand_increase",)).payload.availability
        == "available"
    )  # nosec B101
    toward_power = c.sensitivity_result(
        POINT,
        operator="directional",
        direction=(-1, 0),
        side="forward",
        outputs=("delivered_heat",),
    )
    assert toward_power.payload.derivative.values == (0,)  # nosec B101
    near = c.sensitivity_result({**POINT, "source_derating": 0.1 + 1e-10})
    assert near.payload.availability == "unknown"  # nosec B101
    assert near.execution == "unresolved"  # nosec B101


def test_domain_boundary_and_degenerate_axis():
    c = example_claim()
    assert (
        c.sensitivity_result(c.uncertainty.nominal).payload.availability == "undefined"
    )  # nosec B101
    assert (
        c.sensitivity_result(c.uncertainty.nominal, side="forward").payload.availability
        == "available"
    )  # nosec B101
    zero = c.sensitivity_result(
        c.uncertainty.nominal, operator="directional", direction=(0, 0)
    )
    assert zero.payload.derivative.values == (0, 0, 0, 0)  # nosec B101
    p = c.uncertainty.parameter("source_derating")
    c = replace(
        c,
        uncertainty=replace(
            c.uncertainty,
            parameters=(
                replace(p, nominal=0.1, lower=0.1, upper=0.1),
                c.uncertainty.parameter("demand_increase"),
            ),
        ),
    )
    assert (
        c.sensitivity_result(POINT, side="forward").payload.availability == "undefined"
    )  # nosec B101


@pytest.mark.parametrize(
    "options",
    [
        {"response": "fixed_controls"},
        {"response": "frozen_policy"},
        {"operator": "hessian"},
        {"method": "finite_difference"},
    ],
)
def test_unsupported_sensitivity_has_no_fallback(options):
    result = example_claim().sensitivity_result(POINT, **options)
    assert result.execution == "unsupported"  # nosec B101
    assert result.payload.availability == "not_evaluated"  # nosec B101
    assert result.payload.derivative is None  # nosec B101
    assert not result.supporting_evaluations  # nosec B101


@pytest.mark.parametrize(
    "options",
    [
        {"inputs": ("unknown",)},
        {"inputs": ("source_derating", "source_derating")},
        {"outputs": ()},
        {"coordinate_space": "mystery"},
        {"side": "central"},
        {"direction": (1, 0)},
        {"operator": "directional", "direction": (float("nan"), 0)},
        {"operator": "directional", "direction": (1,)},
    ],
)
def test_invalid_sensitivity(options):
    result = example_claim().sensitivity_result(POINT, **options)
    assert result.execution == "invalid"  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101


def test_storage_conflicts_cardinality_and_shared_policy_axes():
    c = example_storage_claim()
    fixed = c.audit_result("fixed", fixed_preparation_power_mw=0.5)
    causal = c.audit_result()
    perfect = c.audit_result("perfect_foresight")
    assert (fixed.payload.verdict, causal.payload.verdict, perfect.payload.verdict) == (
        "fail",
        "fail",
        "pass",
    )  # nosec B101
    witness = causal.payload.witness
    assert witness.kind == "incompatible_futures"  # nosec B101
    assert witness.cardinality.lower.value == witness.cardinality.upper.value == 2  # nosec B101
    assert fixed.payload.witness.cardinality.upper.value == 1  # nosec B101
    assert causal.payload.search is None  # nosec B101
    infeasible = causal.supporting_evaluations[0]
    assert (
        infeasible.contract.operating["feasibility_scope"] == "model_and_requirements"
    )  # nosec B101
    assert "preparation_charge" not in values(infeasible)  # nosec B101
    assert all(c.assessment == "unknown" for c in infeasible.payload.constraint_checks)  # nosec B101
    assert len(infeasible.contract.quantity("prepared_energy").axes[0].coordinates) == 1  # nosec B101
    assert len(perfect.contract.quantity("prepared_energy").axes[0].coordinates) == 2  # nosec B101
    larger = replace(c, design=StorageDesign(capacity_mwh=2)).audit_result()
    assert larger.payload.verdict == "pass"  # nosec B101
    vals = values(larger.supporting_evaluations[0])
    assert vals["prepared_energy"] == (1,)  # nosec B101
    assert vals["terminal_energy"] == (0, 2)  # nosec B101
    assert vals["total_preparation_charge"] == 1  # nosec B101
    assert larger.contract_ref != causal.contract_ref  # nosec B101
    assert perfect.contract.domain == causal.contract.domain  # nosec B101
    assert perfect.contract_ref != causal.contract_ref  # nosec B101


def test_storage_information_change_is_separate_operating_contract():
    c = example_storage_claim()
    changed = replace(
        c,
        tree=replace(
            c.tree,
            scenarios=tuple(
                replace(s, preparation_observation=s.name) for s in c.tree.scenarios
            ),
        ),
    )
    before, after = c.audit_result(), changed.audit_result()
    assert before.contract.domain == after.contract.domain  # nosec B101
    assert before.contract.operating != after.contract.operating  # nosec B101
    assert after.payload.verdict == "pass"  # nosec B101
    assert changed.sensitivity_result().execution == "unsupported"  # nosec B101


@pytest.mark.parametrize(
    "mode,power",
    [
        ("wrong", None),
        ("fixed", None),
        ("fixed", -1),
        ("fixed", float("nan")),
        ("causal", 0.5),
    ],
)
def test_storage_invalid_contract_not_refuted(mode, power):
    c = example_storage_claim()
    for r in (
        c.evaluate_result(mode, fixed_preparation_power_mw=power),
        c.audit_result(mode, fixed_preparation_power_mw=power),
    ):
        assert r.execution == "invalid"  # nosec B101
        assert result_from_json(r.to_json()) == r  # nosec B101


def test_round_trip_has_no_solver_and_nested_records_are_immutable(monkeypatch):
    c = example_claim()
    results = [
        c.evaluate_result(POINT),
        c.sensitivity_result(POINT),
        c.audit_result(),
        c.breaking_result(),
        example_storage_claim().audit_result(),
        example_storage_claim().audit_result("perfect_foresight"),
    ]

    def no_solver(*args, **kwargs):
        raise AssertionError("loading must not execute solver")

    monkeypatch.setattr(linear, "linprog", no_solver)
    for result in results:
        encoded = result.to_json()
        assert result_from_json(encoded) == result  # nosec B101
        assert type(result).from_json(encoded).result_id == result.result_id  # nosec B101
        with pytest.raises(TypeError):
            result.request["query"] = "changed"
        with pytest.raises(FrozenInstanceError):
            result.execution = "invalid"
        with pytest.raises(TypeError):
            result.contract.design["capacity_mwh"] = 100
        data = result.to_dict()
        data["artifacts"]["contract"]["model_id"] = "changed"
        assert result.contract.model_id != "changed"  # nosec B101


def test_decoder_rejects_tampering_and_missing_references():
    result = example_claim().sensitivity_result(POINT)
    data = result.to_dict()
    data["payload"]["derivative"]["values"][0][0] = -999
    with pytest.raises(ValueError, match="digest"):
        result_from_json(json.dumps(data))
    data = result.to_dict()
    data["artifacts"]["evaluations"] = []
    with pytest.raises(ValueError, match="reference"):
        result_from_json(resign(data))
    data = result.to_dict()
    data["evidence"] = []
    with pytest.raises(ValueError, match="evidence"):
        result_from_json(resign(data))
    data = result.to_dict()
    data["payload"]["derivative"]["units"][0][0] = "GBP"
    with pytest.raises(ValueError, match="units"):
        result_from_json(resign(data))
    data = result.to_dict()
    data["payload"]["derivative"]["values"].pop()
    with pytest.raises(ValueError, match="shape"):
        result_from_json(resign(data))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "process_result/v999"),
        ("kind", "economics"),
        ("execution", "success"),
    ],
)
def test_decoder_rejects_unknown_schema_enums(field, value):
    data = example_claim().evaluate_result(POINT).to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        result_from_json(resign(data))


def test_decoder_rejects_extra_fields_duplicate_keys_nonfinite_numbers():
    result = example_claim().evaluate_result(POINT)
    with pytest.raises(ValueError, match="duplicate"):
        result_from_json('{"kind":"evaluation","kind":"evaluation"}')
    with pytest.raises(ValueError, match="nonfinite"):
        result_from_json('{"n":NaN}')
    data = result.to_dict()
    data["unexpected"] = 0
    with pytest.raises(ValueError, match="envelope"):
        result_from_json(resign(data))
    data = result.to_dict()
    data["payload"]["unexpected"] = 0
    with pytest.raises(ValueError):
        result_from_json(resign(data))
    data = result.to_dict()
    data["payload"]["values"][0]["value"] = 1e999
    with pytest.raises(ValueError):
        result_from_json(json.dumps(data))


def test_constructor_rejects_false_bounds_assessments_and_misalignment():
    c = example_claim()
    evaluation = c.evaluate_result(POINT)
    with pytest.raises(ValueError, match="shape"):
        replace(
            evaluation,
            payload=replace(
                evaluation.payload, values=(QuantityValue("delivered_heat", (7.2,)),)
            ),
        )
    with pytest.raises(ValueError, match="assessment"):
        replace(
            evaluation,
            payload=replace(
                evaluation.payload,
                constraint_checks=(
                    replace(
                        evaluation.payload.constraint_checks[-1], assessment="violated"
                    ),
                ),
            ),
        )
    with pytest.raises(ValueError, match="bounds"):
        replace(
            evaluation.payload.objective, lower=Bound("finite", "MW", 8, ("solver",))
        )
    with pytest.raises(ValueError, match="different contract"):
        replace(
            c.audit_result(),
            supporting_evaluations=(replace(c, cop=5).evaluate_result(POINT),),
        )
    sensitivity = c.sensitivity_result(POINT)
    with pytest.raises(ValueError, match="available"):
        replace(sensitivity.payload, availability="unknown")
    with pytest.raises(ValueError, match="direction"):
        replace(sensitivity.payload.derivative, direction=(1, 0))
    with pytest.raises(ValueError, match="recourse infeasibility"):
        replace(
            evaluation,
            payload=replace(
                evaluation.payload, feasibility="infeasible", objective=None
            ),
        )
    with pytest.raises(ValueError, match="witness"):
        replace(c.audit_result().payload, witness=None)


def test_derivative_type_is_public():
    assert isinstance(
        example_claim().sensitivity_result(POINT).payload.derivative, Derivative
    )  # nosec B101
