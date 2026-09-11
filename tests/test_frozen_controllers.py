"""Frozen commands retain their operating semantics, permissions and evidence."""

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from operange import (
    AffineControlRule,
    AffineController,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    ChangeComparison,
    DecisionRule,
    EngineeringChange,
    FiniteSet,
    FrozenController,
    NormalizedLInf,
    Parameter,
    RecoursePolicy,
    Scenario,
    result_from_json,
)
from operange import linear
from operange.contract_types import reference
from examples.linear_dispatch import example
from tests.test_linear_process import small_model, cases


def allocation(a, b):
    return AffineController(
        f"Allocation {a}/{b}",
        tuple(
            AffineControlRule(
                name,
                "MW",
                0,
                tuple(
                    AffineTerm(observation, share, "MW/MW")
                    for observation in ("dryer", "evaporator")
                ),
            )
            for name, share in (("boiler_a", a), ("boiler_b", b))
        ),
        "Synthetic caller-supplied rule; no fitting",
    )


def single_rule(offset=0, coefficient=1):
    return AffineController(
        "Single control",
        (
            AffineControlRule(
                "power", "MW", offset, (AffineTerm("load", coefficient, "MW/MW"),)
            ),
        ),
        "Synthetic test rule",
    )


@pytest.fixture(scope="module")
def boilers():
    model, domain = example()
    return replace(
        model,
        operating_limits=tuple(
            replace(r, limit=32) if r.name == "shared_fuel" else r
            for r in model.operating_limits
        ),
    ), domain


def values(result):
    return {v.quantity_ref: v.value for v in result.payload.values}


def portable(result):
    for compact in (False, True):
        assert (
            result_from_json(result.to_json(compact=compact)).to_dict()
            == result.to_dict()
        )  # nosec B101


def test_controller_failure_is_not_adjustable_infeasibility(boilers):
    model, domain = boilers
    original = model.as_claim(domain)
    equal = original.with_controller(allocation(0.5, 0.5))
    better = original.with_controller(allocation(0.6, 0.4))
    failure = equal.audit_result()
    assert original.audit_result().payload.verdict == "pass"  # nosec B101
    assert better.audit_result().payload.verdict == "pass"  # nosec B101
    assert failure.payload.verdict == "fail"  # nosec B101
    assert failure.payload.witness.kind == "fixed_policy_failure"  # nosec B101
    assert failure.payload.witness.affected_constraints == ("shared_fuel",)  # nosec B101
    assert "other controllers not assessed" in failure.payload.witness.details["scope"]  # nosec B101
    point = domain.scenario("combined").values
    failed_response, passing_response = (
        equal.evaluate_result(point),
        better.evaluate_result(point),
    )
    assert (
        values(failed_response)["boiler_a"] == values(failed_response)["boiler_b"] == 10
    )  # nosec B101
    assert values(failed_response)["fuel"] == 32.5  # nosec B101
    assert values(passing_response)["fuel"] == 31  # nosec B101
    assert (
        next(
            c
            for c in failed_response.payload.constraint_checks
            if c.constraint_ref == "meet_load"
        ).assessment
        == "satisfied"
    )  # nosec B101
    assert equal.recourse == original.recourse and original.adapter is model  # nosec B101
    assert "controller" not in original.contract.operating  # nosec B101
    for result in (failure, failed_response, passing_response):
        portable(result)


def test_direct_replay_and_audit_never_call_dispatch_solver(boilers, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("controller replay invoked the optimizer")

    monkeypatch.setattr(linear, "solve_lp", forbidden)
    monkeypatch.setattr(linear, "linprog", forbidden)
    model, domain = boilers
    for controller, verdict in (
        (allocation(0.5, 0.5), "fail"),
        (allocation(0.6, 0.4), "pass"),
    ):
        claim = model.as_claim(domain, controller=controller)
        assert claim.audit_result().payload.verdict == verdict  # nosec B101
        assert (
            FrozenController.from_json(claim.freeze().to_json())
            .audit_result()
            .payload.verdict
            == verdict
        )  # nosec B101


def test_frozen_artifact_and_controller_round_trip(boilers):
    model, domain = boilers
    controller = allocation(0.6, 0.4)
    claim = model.as_claim(domain, controller=controller)
    frozen = claim.freeze()
    loaded = FrozenController.from_json(frozen.to_json())
    assert AffineController.from_json(controller.to_json()) == controller  # nosec B101
    assert loaded.fingerprint == frozen.fingerprint  # nosec B101
    assert loaded.to_dict() == frozen.to_dict()  # nosec B101
    assert loaded.model == model and loaded.controller == controller  # nosec B101
    assert loaded.contract.ref == claim.contract.ref  # nosec B101
    assert loaded.audit_result().result_id == claim.audit_result().result_id  # nosec B101
    assert (
        loaded.evaluate_result(domain.scenario("combined").values).result_id
        == claim.evaluate_result(domain.scenario("combined").values).result_id
    )  # nosec B101
    with pytest.raises(FrozenInstanceError):
        loaded.controller.rules[0].offset = 99
    with pytest.raises(TypeError):
        loaded.contract.operating["controller"] = {}


def test_loading_does_not_execute_controller_or_model(boilers, monkeypatch):
    model, domain = boilers
    frozen = model.as_claim(domain, controller=allocation(0.6, 0.4)).freeze()
    document = frozen.to_json()

    def forbidden(*args, **kwargs):
        raise AssertionError("loading executed commands or physical evaluation")

    monkeypatch.setattr(AffineController, "commands", forbidden)
    monkeypatch.setattr(AffineControlRule, "command", forbidden)
    monkeypatch.setattr(linear, "solve_lp", forbidden)
    assert FrozenController.from_json(document).frozen_id == frozen.frozen_id  # nosec B101


def test_new_scenarios_require_explicit_domain_and_keep_same_controller(boilers):
    model, domain = boilers
    frozen = model.as_claim(domain, controller=allocation(0.6, 0.4)).freeze()
    point = {"dryer": 11, "evaporator": 6}
    assert frozen.evaluate_result(point).execution == "invalid"  # nosec B101
    held_out = FiniteSet(domain.space, (Scenario("held out", point, "New test case"),))
    replay = frozen.as_claim(held_out)
    result = replay.audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    assert len(result.payload.coverage.evaluated_support["scenarios"]) == 1  # nosec B101
    assert result.contract.domain != frozen.contract.domain  # nosec B101
    assert replay.adapter.controller.ref == frozen.controller.ref  # nosec B101
    assert replay.freeze().frozen_id != frozen.frozen_id  # nosec B101
    assert frozen.as_claim().domain == domain  # nosec B101
    assert values(replay.evaluate_result(point))["boiler_a"] == pytest.approx(10.2)  # nosec B101


def test_controller_comparison_is_an_operating_change(boilers):
    model, domain = boilers
    baseline = model.as_claim(domain, controller=allocation(0.5, 0.5))
    better = model.as_claim(domain, controller=allocation(0.6, 0.4))
    result = baseline.compare_changes(
        (
            EngineeringChange("60/40", better, "Synthetic controller revision"),
            EngineeringChange(
                "Adjustable", model.as_claim(domain), "Separate feasibility benchmark"
            ),
        )
    )
    assert [c.transition for c in result.candidates] == ["restored", "restored"]  # nosec B101
    assert result.candidates[0].delta.changed_sections == ("operating",)  # nosec B101
    assert (
        result.candidates[0].delta.model_preserved
        and result.candidates[0].delta.requirements_preserved
    )  # nosec B101
    assert result.baseline.payload.witness.kind == "fixed_policy_failure"  # nosec B101
    assert ChangeComparison.from_json(result.to_json(compact=True)).ref == result.ref  # nosec B101


def test_rule_uses_only_permitted_observations(boilers):
    model, domain = boilers
    controller = AffineController(
        "Observe dryer only",
        (
            AffineControlRule("boiler_a", "MW", 0, (AffineTerm("dryer", 1, "MW/MW"),)),
            AffineControlRule("boiler_b", "MW", 6),
        ),
        "Partial observation test",
    )
    claim = model.as_claim(domain, controller=controller)
    first = claim.evaluate_result({"dryer": 10, "evaporator": 6})
    second = claim.evaluate_result({"dryer": 10, "evaporator": 8})
    assert (
        (values(first)["boiler_a"], values(first)["boiler_b"])
        == (values(second)["boiler_a"], values(second)["boiler_b"])
        == (10, 6)
    )  # nosec B101
    assert second.payload.feasibility == "infeasible"  # nosec B101
    execution = next(e for e in second.evidence if e.subject == "controller_execution")
    assert execution.details["observations_by_control"] == {
        "boiler_a": {"dryer": 10},
        "boiler_b": {},
    }  # nosec B101
    assert controller.commands({"dryer": 10, "future": object()}) == {
        "boiler_a": 10,
        "boiler_b": 6,
    }  # nosec B101
    forbidden = replace(
        controller,
        rules=(
            replace(controller.rules[0], terms=(AffineTerm("evaporator", 1, "MW/MW"),)),
            controller.rules[1],
        ),
    )
    with pytest.raises(ValueError, match="outside its permissions"):
        claim.with_controller(forbidden)


@pytest.mark.parametrize("offset", [-0.1, 1.000000000001])
def test_commands_are_never_clipped_to_equipment_bounds(offset):
    model = small_model()
    controller = AffineController(
        "Constant", (AffineControlRule("power", "MW", offset),), "Test command"
    )
    claim = model.as_claim(cases(model, 0), controller=controller)
    result = claim.evaluate_result({"load": 0})
    assert values(result)["power"] == offset  # nosec B101
    assert result.payload.feasibility == "infeasible"  # nosec B101
    assert any(
        c.constraint_ref.startswith("control_") and c.assessment == "violated"
        for c in result.payload.constraint_checks
    )  # nosec B101
    assert claim.recourse.mode == "fixed"  # nosec B101
    portable(result)


def test_selected_requirements_and_operating_limits_survive_freezing(boilers):
    model, domain = boilers
    model = replace(
        model,
        requirements=model.requirements
        + (AffineRequirement("reserve", "steam_balance", 1, "ge"),),
    )
    claim = model.as_claim(
        domain, controller=allocation(0.5, 0.5), requirements=("meet_load",)
    )
    frozen = FrozenController.from_json(claim.freeze().to_json())
    result = frozen.evaluate_result(domain.scenario("combined").values)
    assert frozen.as_claim().requirements == ("meet_load",)  # nosec B101
    assert "shared_fuel" in result.payload.included_constraints  # nosec B101
    assert "reserve" not in result.payload.included_constraints  # nosec B101
    assert result.payload.feasibility == "infeasible"  # nosec B101


def test_overflow_remains_unresolved_and_does_not_hide_verified_failure():
    model = small_model()
    claim = model.as_claim(
        cases(model, 0, 1, 2), controller=single_rule(coefficient=1e308)
    )
    result = claim.audit_result()
    assert result.execution == "unresolved" and result.payload.verdict == "fail"  # nosec B101
    assert result.payload.coverage.method == "partial"  # nosec B101
    assert result.payload.witness.realizations == ({"load": 1},)  # nosec B101
    assert result.payload.coverage.unexplored_support["unresolved_scenarios"] == ("2",)  # nosec B101
    overflow = claim.evaluate_result({"load": 2})
    assert overflow.payload.feasibility == "unknown"  # nosec B101
    assert not any(e.subject == "recourse_infeasibility" for e in overflow.evidence)  # nosec B101
    portable(result)
    portable(overflow)


def test_command_underflow_checks_the_actual_rounded_command():
    model = small_model(tolerance=0)
    claim = model.as_claim(
        cases(model, 1e-308), controller=single_rule(coefficient=1e-308)
    )
    result = claim.evaluate_result({"load": 1e-308})
    assert values(result)["power"] == 0  # nosec B101
    assert result.payload.feasibility == "infeasible"  # nosec B101


def test_point_evaluation_does_not_grant_continuous_audits_or_distances():
    model = small_model()
    domain = BoxSet((Parameter("load", "MW", 0.5, 0, 1, 0.5, "Test box"),))
    claim = model.as_claim(domain, controller=single_rule())
    assert claim.evaluate_result({"load": 0.25}).payload.feasibility == "feasible"  # nosec B101
    for result in (
        claim.audit_result(),
        claim.sensitivity_result({"load": 0.25}),
        claim.boundary_result(),
        claim.breaking_result(),
    ):
        assert result.execution == "unsupported"  # nosec B101
    metric_claim = replace(claim, distance=NormalizedLInf(domain.space))
    loaded = FrozenController.from_json(metric_claim.freeze().to_json()).as_claim()
    assert loaded.distance == metric_claim.distance  # nosec B101
    assert (
        loaded.evaluate_result({"load": 0.25}).result_id
        == claim.evaluate_result({"load": 0.25}).result_id
    )  # nosec B101


@pytest.mark.parametrize(
    "point",
    [
        None,
        {},
        {"load": True},
        {"load": float("nan")},
        {"load": float("inf")},
        {"future": 1},
        {"load": 1, "future": 0},
    ],
)
def test_invalid_observations_never_produce_commands(point):
    model = small_model()
    result = model.as_claim(cases(model, 1), controller=single_rule()).evaluate_result(
        point
    )
    assert (
        result.execution == "invalid" and result.payload.feasibility == "not_evaluated"
    )  # nosec B101
    assert result.payload.values == ()  # nosec B101
    portable(result)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: replace(c, rules=(replace(c.rules[0], control="unknown"),)),
        lambda c: replace(c, rules=(replace(c.rules[0], unit="kW"),)),
        lambda c: replace(c, rules=(replace(c.rules[0], stage="preparation"),)),
        lambda c: replace(
            c, rules=(replace(c.rules[0], terms=(AffineTerm("future", 1, "MW/MW"),)),)
        ),
        lambda c: replace(
            c, rules=(replace(c.rules[0], terms=(AffineTerm("power", 1, "MW/MW"),)),)
        ),
        lambda c: replace(
            c, rules=(replace(c.rules[0], terms=(AffineTerm("load", 1, "1"),)),)
        ),
    ],
)
def test_invalid_model_controller_bindings_are_rejected(mutation):
    model = small_model()
    with pytest.raises(ValueError):
        model.as_claim(cases(model, 0), controller=mutation(single_rule()))


def test_frozen_rules_cannot_override_fixed_or_future_permissions():
    model = small_model()
    for policy in (
        RecoursePolicy(
            "fixed", (DecisionRule("power", "MW", "operation", fixed_value=0.5),)
        ),
        RecoursePolicy(
            "causal", (DecisionRule("power", "MW", "operation", ("load",)),)
        ),
        RecoursePolicy(
            "static", (DecisionRule("power", "MW", "operation", ("future",)),)
        ),
        RecoursePolicy("fixed", ()),
    ):
        with pytest.raises(ValueError):
            model.as_claim(cases(model, 0), recourse=policy, controller=single_rule())
    with pytest.raises(ValueError):
        model.as_claim(cases(model, 0), recourse={}, controller=single_rule())


def test_rule_declarations_and_identity_validation():
    for invalid in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            AffineControlRule("power", "MW", invalid)
    controller = single_rule()
    for data in (
        {"name": " "},
        {"provenance": ""},
        {"rules": ()},
        {"rules": controller.rules * 2},
    ):
        with pytest.raises(ValueError):
            replace(controller, **data)
    with pytest.raises(ValueError):
        replace(controller.rules[0], terms=controller.rules[0].terms * 2)
    with pytest.raises(ValueError):
        controller.commands({})
    assert single_rule(coefficient=0.5).ref != controller.ref  # nosec B101
    assert replace(controller, provenance="Different source").ref != controller.ref  # nosec B101
    data = controller.to_manifest()
    data["controller"]["rules"][0]["offset"] = 1
    with pytest.raises(ValueError, match="identity"):
        AffineController.from_json(json.dumps(data))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(frozen_id="sha256:" + "0" * 64),
        lambda d: d.update(schema_version="frozen_controller/v2"),
        lambda d: d.update(extra="ignored"),
        lambda d: d["contract"].update(model_id="consumer_callback/v1"),
        lambda d: d["contract"]["model"]["controls"][0].update(upper=999),
        lambda d: d["contract"]["operating"]["controller"]["controller"]["rules"][
            0
        ].update(offset=99),
        lambda d: d["contract"]["operating"].update(recourse_policy={}),
        lambda d: d["contract"]["numerical_policy"].update(controller_clipping=True),
        lambda d: d["contract"]["domain"].update(extra=True),
        lambda d: d.update(distance={"kind": "unknown", "space": {}}),
    ],
)
def test_frozen_loading_rejects_tampered_artifacts(boilers, mutation):
    model, domain = boilers
    data = model.as_claim(domain, controller=allocation(0.6, 0.4)).freeze().to_dict()
    mutation(data)
    with pytest.raises(ValueError):
        FrozenController.from_json(json.dumps(data))


def test_rehashed_inconsistent_executable_binding_is_rejected(boilers):
    model, domain = boilers
    data = model.as_claim(domain, controller=allocation(0.6, 0.4)).freeze().to_dict()
    data["contract"]["operating"]["response"] = "arbitrary_python_callback"
    data["frozen_id"] = reference(
        {k: v for k, v in data.items() if k != "frozen_id"}, "frozen_controller/v1"
    ).artifact_id
    with pytest.raises(ValueError, match="does not match"):
        FrozenController.from_json(json.dumps(data))


@pytest.mark.parametrize(
    "reader", [AffineController.from_json, FrozenController.from_json]
)
@pytest.mark.parametrize(
    "document", ["{}", "null", "[]", '{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}']
)
def test_controller_readers_reject_invalid_json(reader, document):
    with pytest.raises(ValueError):
        reader(document)


def test_freeze_requires_an_executable_supported_controller(boilers):
    from examples.steam_header import example as affine_example

    model, domain = boilers
    with pytest.raises(ValueError, match="explicit supported controller"):
        model.as_claim(domain).freeze()
    affine, box = affine_example()
    with pytest.raises(ValueError, match="require LinearProcessAdapter"):
        affine.as_claim(box).with_controller(allocation(0.6, 0.4))


def test_freeze_rejects_custom_domain_implementations_including_nested_factors():
    from operange import Union

    class ConsumerDomain(FiniteSet):
        pass

    model = small_model()
    declared = cases(model, 0)
    custom = ConsumerDomain(declared.space, declared.scenarios)
    nested = Union((custom,), feasible_point={"load": 0}, provenance="Test composition")
    for domain in (custom, nested):
        with pytest.raises(ValueError, match="supported declarative domain"):
            model.as_claim(domain, controller=single_rule()).freeze()
    supported = Union(
        (declared,), feasible_point={"load": 0}, provenance="Test composition"
    )
    frozen = model.as_claim(supported, controller=single_rule()).freeze()
    assert (
        FrozenController.from_json(frozen.to_json()).fingerprint == frozen.fingerprint
    )  # nosec B101


def test_standalone_consumer_example_replays_after_export():
    from examples.frozen_controllers import run_example

    output = run_example()
    comparison = ChangeComparison.from_json(json.dumps(output["comparison"]))
    frozen = FrozenController.from_json(json.dumps(output["frozen_controller"]))
    replay = result_from_json(json.dumps(output["replayed_audit"]))
    held_out = result_from_json(json.dumps(output["held_out_audit"]))
    assert [c.transition for c in comparison.candidates] == ["restored", "restored"]  # nosec B101
    assert frozen.audit_result().result_id == replay.result_id  # nosec B101
    assert held_out.payload.verdict == "pass"  # nosec B101
    assert held_out.contract.domain != replay.contract.domain  # nosec B101
