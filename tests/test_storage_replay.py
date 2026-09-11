"""Causal execution, physical state accounting, and portable storage replay."""

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from operange import EngineeringChange, FrozenController, result_from_json
from operange.contract_types import reference
from operange.reference import (
    ScenarioTree,
    StorageController,
    StorageDesign,
    StoragePreparationRule,
    StorageRequirement,
    StorageScenario,
    ThermalStorageClaim,
    example_storage_claim,
)


def controller(charge=1, *, default=None):
    return StorageController(
        "Preparation",
        (StoragePreparationRule("unrevealed", charge),),
        "Synthetic prescribed rule",
        default,
    )


def model():
    return replace(example_storage_claim(), design=StorageDesign(capacity_mwh=2))


def paths(result):
    return next(
        e.details["paths"] for e in result.evidence if e.evidence_id == "replay"
    )


def values(result):
    return {v.quantity_ref: v.value for v in result.payload.values}


def portable(result):
    for compact in (False, True):
        assert (
            result_from_json(result.to_json(compact=compact)).to_dict()
            == result.to_dict()
        )  # nosec B101


def test_shared_history_has_identical_preparation_and_carried_state():
    result = model().as_claim(controller=controller()).audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    assert result.payload.coverage.method == "complete_finite"  # nosec B101
    rows = paths(result)
    assert [p["scenario"]["name"] for p in rows] == ["Demand surge", "Heat surplus"]  # nosec B101
    assert rows[0]["steps"][0] == rows[1]["steps"][0]  # nosec B101
    for row, terminal in zip(rows, (0, 2)):
        prep, event = row["steps"]
        assert prep["observations"] == {"preparation_observation": "unrevealed"}  # nosec B101
        assert prep["state_before_mwh"] == 0  # nosec B101
        assert prep["state_after_mwh"] == event["state_before_mwh"] == 1  # nosec B101
        assert event["state_after_mwh"] == terminal  # nosec B101
    assert result.supporting_evaluations[0].payload.objective is None  # nosec B101
    portable(result)


def test_no_solver_or_redispatch_and_failure_does_not_refute_other_controllers(
    monkeypatch,
):
    from operange import storage, linear

    plant = model()
    assert plant.audit().status == "pass"  # nosec B101

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not optimize")

    monkeypatch.setattr(storage, "solve_lp", forbidden)
    monkeypatch.setattr(linear, "solve_lp", forbidden)
    monkeypatch.setattr(ThermalStorageClaim, "audit", forbidden)
    frozen = plant.as_claim(controller=controller(0.5)).freeze()
    result = FrozenController.from_json(frozen.to_json()).audit_result()
    assert result.payload.verdict == "fail"  # nosec B101
    assert result.payload.witness.kind == "fixed_policy_failure"  # nosec B101
    assert "terminal_min" in result.payload.witness.affected_constraints  # nosec B101
    assert paths(result)[0]["steps"][1]["state_after_mwh"] == -0.5  # nosec B101
    assert "other controllers not assessed" in result.payload.witness.details["scope"]  # nosec B101
    assert (
        plant.as_claim(controller=controller()).audit_result().payload.verdict == "pass"
    )  # nosec B101
    portable(result)


def test_roundtrip_preserves_full_model_and_load_does_not_execute(monkeypatch):
    plant = replace(
        model(),
        name="Original name",
        tolerance=2e-8,
        requirement=StorageRequirement(0.25, 0.1, 1.9),
    )
    frozen = plant.as_claim(controller=controller(0.75)).freeze()
    expected = frozen.audit_result()

    def forbidden(*args, **kwargs):
        raise AssertionError("Loading must not execute")

    with monkeypatch.context() as patch:
        patch.setattr(StorageController, "preparation_command", forbidden)
        patch.setattr(StorageController, "event_commands", forbidden)
        patch.setattr(ThermalStorageClaim, "audit", forbidden)
        loaded = FrozenController.from_json(frozen.to_json())
    assert loaded.model == plant  # nosec B101
    assert loaded.controller == frozen.controller  # nosec B101
    assert loaded.to_dict() == frozen.to_dict()  # nosec B101
    assert loaded.audit_result().result_id == expected.result_id  # nosec B101
    assert StorageController.from_json(frozen.controller.to_json()) == frozen.controller  # nosec B101
    with pytest.raises(FrozenInstanceError):
        loaded.controller.default_preparation_charge_mw = 2


def test_held_out_tree_uses_unchanged_controller_and_explicit_initial_state():
    frozen = model().as_claim(controller=controller()).freeze()
    original = frozen.to_json()
    tree = ScenarioTree(
        (
            StorageScenario("Later demand", 1.5, 0, "held out"),
            StorageScenario("Later recovery", 0, 0.5, "held out"),
        )
    )
    held = frozen.as_claim(tree=tree)
    assert held.adapter.controller is frozen.controller  # nosec B101
    result = held.audit_result()
    assert result.payload.verdict == "fail"  # nosec B101
    assert [p["steps"][0]["state_before_mwh"] for p in paths(result)] == [0, 0]  # nosec B101
    assert frozen.to_json() == original  # nosec B101
    assert held.contract.ref != frozen.contract.ref  # nosec B101
    assert held.freeze().controller.controller_id == frozen.controller.controller_id  # nosec B101
    assert (
        FrozenController.from_json(held.freeze().to_json()).audit_result().result_id
        == result.result_id
    )  # nosec B101
    reversed_result = frozen.as_claim(
        tree=replace(tree, scenarios=tuple(reversed(tree.scenarios)))
    ).audit_result()
    assert [p["scenario"]["name"] for p in paths(reversed_result)] == [
        "Later recovery",
        "Later demand",
    ]  # nosec B101
    assert {p["scenario"]["name"]: p["steps"] for p in paths(result)} == {
        p["scenario"]["name"]: p["steps"] for p in paths(reversed_result)
    }  # nosec B101


def test_signal_table_is_not_indexed_by_future_identity():
    plant = example_storage_claim()
    table = StorageController(
        "Signals",
        (
            StoragePreparationRule("demand signal", 1),
            StoragePreparationRule("surplus signal", 0),
        ),
        "Synthetic earlier information",
    )
    tree = replace(
        plant.tree,
        scenarios=tuple(
            replace(s, preparation_observation=signal)
            for s, signal in zip(
                plant.tree.scenarios, ("demand signal", "surplus signal")
            )
        ),
    )
    known = replace(plant, tree=tree).as_claim(controller=table).freeze()
    assert known.audit_result().payload.verdict == "pass"  # nosec B101
    hidden = known.as_claim(
        tree=replace(
            tree,
            scenarios=tuple(
                replace(s, preparation_observation="demand signal")
                for s in tree.scenarios
            ),
        )
    ).audit_result()
    assert hidden.payload.verdict == "fail"  # nosec B101
    assert paths(hidden)[0]["steps"][0] == paths(hidden)[1]["steps"][0]  # nosec B101
    # Scenario names and future physical values cannot override a shared signal.
    assert paths(hidden)[1]["steps"][0]["commands"]["preparation_charge"] == 1  # nosec B101


@pytest.mark.parametrize("known_charge, verdict", [(1, "inconclusive"), (0.5, "fail")])
def test_unknown_signal_is_partial_without_erasing_verified_failure(
    known_charge, verdict
):
    plant = model()
    unknown = replace(plant.tree.scenarios[1], preparation_observation="unseen")
    plant = replace(
        plant, tree=replace(plant.tree, scenarios=(unknown, plant.tree.scenarios[0]))
    )
    result = plant.as_claim(controller=controller(known_charge)).audit_result()
    assert result.execution == "unresolved"  # nosec B101
    assert result.payload.verdict == verdict  # nosec B101
    assert result.payload.coverage.method == "partial"  # nosec B101
    assert result.payload.coverage.unexplored_support["unresolved_paths"] == (
        "Heat surplus",
    )  # nosec B101
    assert "commands" not in paths(result)[0]["steps"][0]  # nosec B101
    assert len(paths(result)[0]["steps"]) == 1  # nosec B101
    portable(result)
    defaulted = plant.as_claim(
        controller=controller(known_charge, default=0)
    ).audit_result()
    assert defaulted.execution == "completed"  # nosec B101
    assert paths(defaulted)[0]["steps"][0]["commands"]["preparation_charge"] == 0  # nosec B101


def test_empty_table_with_explicit_default():
    rule = StorageController("Constant", (), "Declared fallback", 1)
    assert model().as_claim(controller=rule).audit_result().payload.verdict == "pass"  # nosec B101
    with pytest.raises(ValueError, match="rule or explicit default"):
        replace(rule, default_preparation_charge_mw=None)


@pytest.mark.parametrize(
    "charge, affected",
    [(-0.5, "preparation_charge_lower"), (1.5, "preparation_charge_upper")],
)
def test_commands_are_not_clipped(charge, affected):
    result = model().as_claim(controller=controller(charge)).evaluate_result()
    assert result.payload.feasibility == "infeasible"  # nosec B101
    assert values(result)["preparation_charge"] == (charge,)  # nosec B101
    assert any(
        c.constraint_ref == affected and c.assessment == "violated"
        for c in result.payload.constraint_checks
    )  # nosec B101
    portable(result)


def test_event_commands_remain_unclipped_by_power_or_available_energy():
    plant = replace(
        model(), tree=ScenarioTree((StorageScenario("High demand", 3, 0, "test"),))
    )
    result = plant.as_claim(controller=controller()).audit_result()
    event = paths(result)[0]["steps"][1]
    assert event["commands"] == {"event_charge": 0, "event_discharge": 3}  # nosec B101
    assert event["state_after_mwh"] == -2  # nosec B101
    assert "event_discharge_upper" in result.payload.witness.affected_constraints  # nosec B101


def test_losses_periods_efficiencies_initial_and_terminal_state():
    plant = ThermalStorageClaim(
        "Energy accounting",
        StorageDesign(4, 2, 2, 0.8, 0.5, 0.1),
        ScenarioTree(
            (
                StorageScenario("Demand", 0.5, 0, "test"),
                StorageScenario("Surplus", 0, 0.25, "test"),
            ),
            2,
            2,
        ),
        StorageRequirement(0.5, 0.2, 3),
    )
    frozen = plant.as_claim(controller=controller(1.46875)).freeze()
    result = FrozenController.from_json(frozen.to_json()).audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    for path, expected in zip(paths(result), (0.2, 2.6)):
        prep, event = path["steps"]
        assert prep["state_after_mwh"] == pytest.approx(2.75)  # nosec B101
        assert event["state_before_mwh"] == prep["state_after_mwh"]  # nosec B101
        assert event["state_after_mwh"] == pytest.approx(expected)  # nosec B101
    limited = replace(
        plant, requirement=replace(plant.requirement, terminal_max_mwh=2.5)
    )
    failed = limited.as_claim(controller=frozen.controller).audit_result()
    assert failed.payload.witness.affected_constraints == ("terminal_max",)  # nosec B101


def test_known_command_failure_survives_state_overflow():
    plant = replace(model(), tree=replace(model().tree, preparation_hours=2))
    result = plant.as_claim(controller=controller(1e308)).audit_result()
    assert result.execution == "unresolved"  # nosec B101
    assert result.payload.verdict == "fail"  # nosec B101
    assert result.payload.witness.affected_constraints == ("preparation_charge_upper",)  # nosec B101
    assert paths(result)[0]["steps"][0]["commands"]["preparation_charge"] == 1e308  # nosec B101
    portable(result)


def test_controller_revision_integrates_with_common_comparison():
    plant = model()
    baseline = plant.as_claim(controller=controller(0.5))
    comparison = baseline.compare_changes(
        (
            EngineeringChange(
                "Prepare enough",
                plant.as_claim(controller=controller()),
                "Synthetic rule correction",
            ),
        )
    )
    candidate = comparison.candidates[0]
    assert candidate.transition == "restored"  # nosec B101
    assert candidate.delta.changed_sections == ("operating",)  # nosec B101
    assert candidate.delta.commitment == "preserved"  # nosec B101


@pytest.mark.parametrize(
    "mode, kwargs",
    [("perfect_foresight", {}), ("fixed", {"fixed_preparation_power_mw": 1})],
)
def test_noncausal_permissions_are_rejected(mode, kwargs):
    with pytest.raises(ValueError, match="causal observation permissions"):
        model().as_claim(mode, controller=controller(), **kwargs)


def test_changed_permissions_requirements_and_bare_domain_cannot_be_frozen():
    frozen = model().as_claim(controller=controller()).freeze()
    permission = frozen.claim.recourse
    wrong = replace(permission.rules[0], observes=("event_identity",))
    invalid = replace(
        frozen.claim, recourse=replace(permission, rules=(wrong, *permission.rules[1:]))
    )
    assert invalid.audit_result().execution == "unsupported"  # nosec B101
    with pytest.raises(ValueError, match="causal observation"):
        invalid.freeze()
    reduced = replace(frozen.claim, requirements=("event_service",))
    assert reduced.audit_result().execution == "unsupported"  # nosec B101
    with pytest.raises(ValueError, match="all service and terminal"):
        reduced.freeze()
    new_tree = ScenarioTree((StorageScenario("new", 0.5, 0, "test"),))
    bare = frozen.as_claim(new_tree.domain)
    assert bare.audit_result().execution == "unsupported"  # nosec B101
    with pytest.raises(ValueError, match="bound tree"):
        bare.freeze()
    with pytest.raises(ValueError, match="no domain argument"):
        frozen.as_claim(new_tree.domain, tree=new_tree)


def test_query_scope_rejections():
    claim = model().as_claim(controller=controller())
    assert (
        claim.evaluate_result({"demand": 1, "recovered_heat": 0}).execution == "invalid"
    )  # nosec B101
    assert claim.audit_result(retune=True).execution == "invalid"  # nosec B101
    for result in (
        claim.sensitivity_result(),
        claim.boundary_result(),
        claim.breaking_result(),
    ):
        assert result.execution == "unsupported"  # nosec B101
    with pytest.raises(ValueError, match="explicit supported controller"):
        model().as_claim().freeze()


@pytest.mark.parametrize(
    "field, value",
    [
        ("event_rule", "future_optimizer"),
        ("name", ""),
        ("default_preparation_charge_mw", float("inf")),
        ("preparation_rules", [{"observation": "x", "charge_mw": True}]),
    ],
)
def test_invalid_controller_declarations(field, value):
    with pytest.raises(ValueError):
        replace(controller(), **{field: value})


def test_duplicate_signals_and_custom_rules_rejected():
    with pytest.raises(ValueError, match="duplicate signals"):
        replace(
            controller(),
            preparation_rules=(
                StoragePreparationRule("x", 1),
                StoragePreparationRule("x", 0),
            ),
        )

    class CustomRule(StoragePreparationRule):
        pass

    with pytest.raises(ValueError, match="declarative rules"):
        replace(controller(), preparation_rules=(CustomRule("x", 1),))


@pytest.mark.parametrize(
    "field, value",
    [("default_preparation_charge_mw", 0), ("event_rule", "future_optimizer")],
)
def test_controller_tampering_rejected(field, value):
    data = json.loads(controller().to_json())
    data["controller"][field] = value
    with pytest.raises(ValueError):
        StorageController.from_json(json.dumps(data))


@pytest.mark.parametrize(
    "change",
    [
        "initial",
        "duration",
        "signal",
        "order",
        "policy",
        "numerical",
        "design",
        "model",
        "identity",
    ],
)
def test_frozen_tampering_rejected(change):
    data = model().as_claim(controller=controller()).freeze().to_dict()
    contract = data["contract"]
    if change == "initial":
        contract["model"]["requirement"]["initial_energy_mwh"] = 0.2
    elif change == "duration":
        contract["operating"]["replay_tree"]["event_hours"] = 2
    elif change == "signal":
        contract["operating"]["replay_tree"]["scenarios"][0][
            "preparation_observation"
        ] = "unseen"
    elif change == "order":
        contract["operating"]["replay_tree"]["scenarios"].reverse()
    elif change == "policy":
        contract["operating"]["recourse_policy"]["rules"][0]["observes"] = [
            "event_identity"
        ]
    elif change == "numerical":
        contract["numerical_policy"]["controller_clipping"] = True
    elif change == "design":
        contract["design"]["capacity_mwh"] = 3
    elif change == "model":
        contract["model_id"] = "unavailable_storage/v1"
    else:
        data["frozen_id"] = "wrong"
    with pytest.raises(ValueError):
        FrozenController.from_json(json.dumps(data))


def test_rehashed_inconsistent_manifest_is_not_an_executable_contract():
    data = model().as_claim(controller=controller()).freeze().to_dict()
    data["contract"]["model"]["event_hours"] = 7
    body = {k: v for k, v in data.items() if k != "frozen_id"}
    data["frozen_id"] = reference(body, "frozen_controller/v1").artifact_id
    with pytest.raises(ValueError, match="does not match"):
        FrozenController.from_json(json.dumps(data))


@pytest.mark.parametrize(
    "reader", [StorageController.from_json, FrozenController.from_json]
)
@pytest.mark.parametrize(
    "document", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "{}"]
)
def test_malformed_documents(reader, document):
    with pytest.raises(ValueError):
        reader(document)


def test_custom_physical_implementation_rejected():
    class CustomTree(ScenarioTree):
        pass

    plant = replace(model(), tree=CustomTree(model().tree.scenarios))
    with pytest.raises(ValueError, match="declarative tree"):
        plant.as_claim(controller=controller())


def test_service_residual_checks_actual_rounded_command_without_cancellation():
    plant = ThermalStorageClaim(
        "Large powers",
        StorageDesign(2e16, 1e16, 1e16),
        ScenarioTree((StorageScenario("Cancellation", 1e16, 1, "test"),)),
        StorageRequirement(initial_energy_mwh=1e16),
    )
    result = plant.as_claim(controller=controller(0)).audit_result()
    assert result.payload.verdict == "fail"  # nosec B101
    assert result.payload.witness.affected_constraints == ("event_service",)  # nosec B101
    event = paths(result)[0]["steps"][1]
    assert event["commands"]["event_discharge"] == 1e16  # nosec B101
    assert event["residuals"]["event_service"] == -1  # nosec B101


def test_consumer_example():
    from operange import ChangeComparison
    from examples.storage_replay import run_example

    output = run_example()
    comparison = ChangeComparison.from_json(json.dumps(output["comparison"]))
    assert [c.transition for c in comparison.candidates] == ["restored", "restored"]  # nosec B101
    frozen = FrozenController.from_json(json.dumps(output["frozen_controller"]))
    replay = result_from_json(json.dumps(output["replayed_audit"]))
    assert frozen.audit_result().result_id == replay.result_id  # nosec B101
    held_out = result_from_json(json.dumps(output["held_out_audit"]))
    assert held_out.payload.verdict == "fail"  # nosec B101
    assert held_out.payload.witness.realizations[0]["name"] == "Demand beyond reserve"  # nosec B101
    unknown = result_from_json(json.dumps(output["unknown_signal_audit"]))
    assert unknown.payload.verdict == "inconclusive"  # nosec B101
