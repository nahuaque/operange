"""Common changes retain engineering commitments, evidence and unresolved rows."""

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from operange import (
    ChangeComparison,
    ChangeResult,
    Claim,
    DecisionRule,
    EngineeringChange,
    FiniteSet,
    NormalizedLInf,
    RecoursePolicy,
    compare_changes,
    rejected_result,
)
from operange._change_contracts import compare_contracts
from operange.changes import _requirements
from operange.contract_types import plain
from examples.engineering_changes import example, run_example
from examples.linear_dispatch import example as dispatch_example
from examples.steam_header import example as affine_example


def change(claim, name="proposal"):
    return EngineeringChange(name, claim, "Synthetic test proposal")


@pytest.fixture(scope="module")
def comparison():
    return run_example()


def test_comparison_keeps_order_baseline_and_full_audits(comparison):
    assert comparison.baseline.payload.verdict == "fail"  # nosec B101
    assert [c.name for c in comparison.candidates] == [  # nosec B101
        "Larger boiler",
        "More fuel",
        "Exclude peak",
        "Reduce service",
    ]
    assert [c.audit.payload.verdict for c in comparison.candidates] == [  # nosec B101
        "pass",
        "fail",
        "pass",
        "pass",
    ]
    assert [c.transition for c in comparison.candidates] == [  # nosec B101
        "restored",
        "still_failing",
        "not_comparable",
        "not_comparable",
    ]
    assert comparison.candidates[0].delta.changed_sections == ("design",)  # nosec B101
    assert comparison.candidates[2].delta.domain_preserved is False  # nosec B101
    assert comparison.candidates[3].delta.requirements_preserved is False  # nosec B101
    assert all(c.baseline is comparison.baseline for c in comparison.candidates)  # nosec B101
    assert all(
        c.audit.payload.coverage.method == "complete_finite"  # nosec B101
        for c in comparison.candidates
    )
    assert comparison.baseline.payload.witness is not None  # nosec B101
    with pytest.raises(FrozenInstanceError):
        comparison.candidates[0].name = "changed"
    with pytest.raises(TypeError):
        comparison.candidates[0].audit.contract.design["new"] = 1


def test_replay_original_failure_after_comparison(comparison):
    model, loads = example()
    claim = model.as_claim(loads)
    before = claim.contract.to_dict()
    witness = comparison.baseline.payload.witness
    result = claim.evaluate_result(witness.realizations[0])
    assert result.payload.feasibility == "infeasible"  # nosec B101
    assert claim.contract.to_dict() == before  # nosec B101
    assert model.controls[0].upper == 18  # nosec B101


def test_reverse_comparison_and_no_change(comparison):
    before, after = comparison.baseline, comparison.candidates[0].audit
    assert ChangeResult("smaller", "test", after, before).transition == "regressed"  # nosec B101
    assert ChangeResult("same", "test", after, after).transition == "still_passing"  # nosec B101


def test_observed_margins_do_not_invent_dispatch_at_failed_case(comparison):
    summary = comparison.requirements[0]
    assert summary.unit == "MW" and summary.relation == "le"  # nosec B101
    assert summary.satisfied == 1 and summary.unknown == 1  # nosec B101
    assert summary.violated == 0 and 0 <= summary.observed_margin <= 2  # nosec B101
    assert len(summary.evaluation_refs) == 1  # nosec B101
    model, loads = example()
    failed = model.as_claim(FiniteSet(loads.space, loads.scenarios[1:])).audit_result()
    assert _requirements(failed)[0].observed_margin is None  # nosec B101
    assert _requirements(failed)[0].evaluation_refs == ()  # nosec B101
    assert failed.payload.witness is not None  # nosec B101


@pytest.mark.parametrize("compact", [False, True])
def test_portable_comparison_and_audit_deduplication(comparison, compact):
    restored = ChangeComparison.from_json(comparison.to_json(compact=compact))
    assert restored.to_dict() == comparison.to_dict()  # nosec B101
    assert restored.comparison_id == comparison.comparison_id  # nosec B101
    assert restored.baseline.ref == comparison.baseline.ref  # nosec B101
    same = ChangeComparison(
        comparison.baseline,
        (ChangeResult("same", "test", comparison.baseline, comparison.baseline),),
    )
    assert len(same.to_dict(compact=compact)["artifacts"]["audits"]) == 1  # nosec B101
    assert ChangeComparison.from_json(same.to_json(compact=compact)).ref == same.ref  # nosec B101


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["candidates"][0].update(transition="still_failing"),
        lambda d: d["candidates"][2]["delta"].update(commitment="preserved"),
        lambda d: d["candidates"][0]["delta"].update(differences=[]),
        lambda d: d["candidates"][0]["requirements"][0].update(observed_margin=100),
        lambda d: d["requirements"][0].update(satisfied=True),
        lambda d: d["candidates"][0].update(extra="unrecognized"),
        lambda d: d.update(extra="unrecognized"),
        lambda d: d.update(comparison_id="sha256:" + "0" * 64),
        lambda d: d["baseline_ref"].update(schema_version="other/v1"),
        lambda d: d["artifacts"].update(extra={}),
        lambda d: d["artifacts"]["audits"].pop(d["baseline_ref"]["artifact_id"]),
        lambda d: d.update(schema_version="engineering_change_comparison/v2"),
        lambda d: d["candidates"].append(d["candidates"][0]),
    ],
)
def test_loading_rejects_tampering(comparison, mutation):
    data = comparison.to_dict()
    mutation(data)
    with pytest.raises(ValueError):
        ChangeComparison.from_json(json.dumps(data))


def test_loading_rejects_unused_audit_and_malformed_json(comparison):
    data = comparison.to_dict()
    extra = replace(comparison.baseline, request={"query": "audit", "extra": True})
    data["artifacts"]["audits"][extra.result_id] = extra.to_dict()
    with pytest.raises(ValueError, match="unused"):
        ChangeComparison.from_json(json.dumps(data))
    for text in (
        '{"schema_version":1,"schema_version":2}',
        '{"value": NaN}',
        '{"value": Infinity}',
        "[]",
        "{}",
        "null",
    ):
        with pytest.raises(ValueError):
            ChangeComparison.from_json(text)


def test_loading_does_not_execute_models(comparison, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("loading executed an adapter")

    monkeypatch.setattr(Claim, "audit_result", fail)
    assert ChangeComparison.from_json(comparison.to_json()).ref == comparison.ref  # nosec B101


def test_each_candidate_is_audited_once_and_empty_input_works(monkeypatch):
    model, loads = example()
    claim = model.as_claim(loads)
    original, calls = Claim.audit_result, []

    def audited(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(Claim, "audit_result", audited)
    result = compare_changes(claim, (change(claim, n) for n in ("a", "b")))
    assert len(calls) == 3  # nosec B101
    assert len(result.candidates) == 2  # nosec B101
    empty = claim.compare_changes(())
    assert len(calls) == 4 and empty.candidates == ()  # nosec B101
    assert ChangeComparison.from_json(empty.to_json()).ref == empty.ref  # nosec B101


def test_all_inputs_validated_before_auditing(monkeypatch):
    model, loads = example()
    claim = model.as_claim(loads)

    def fail(self):
        raise AssertionError("invalid inputs executed an audit")

    monkeypatch.setattr(Claim, "audit_result", fail)
    for candidates in ((change(claim), change(claim)), (change(claim), claim)):
        with pytest.raises(ValueError):
            compare_changes(claim, candidates)
    with pytest.raises(ValueError):
        compare_changes(model, ())
    for name, candidate, provenance in (
        (" ", claim, "test"),
        ("a", claim, ""),
        (1, claim, "test"),
        ("a", model, "test"),
    ):
        with pytest.raises(ValueError):
            EngineeringChange(name, candidate, provenance)


def test_adapter_exception_retains_other_rows(monkeypatch):
    model, loads = example()
    baseline = model.as_claim(loads)
    broken = model.as_claim(loads)
    original = Claim.audit_result

    def audited(self):
        if self is broken:
            raise RuntimeError("solver unavailable")
        return original(self)

    monkeypatch.setattr(Claim, "audit_result", audited)
    result = baseline.compare_changes((change(broken), change(baseline, "still runs")))
    first, second = result.candidates
    assert first.audit.execution == "unresolved"  # nosec B101
    assert first.audit.payload.verdict == "not_assessed"  # nosec B101
    assert first.audit.diagnostics[0].code == "comparison_audit_error"  # nosec B101
    assert first.transition == "inconclusive"  # nosec B101
    assert second.transition == "still_failing"  # nosec B101


def test_unresolved_and_unsupported_are_not_restorations(comparison):
    before, passed = comparison.baseline, comparison.candidates[0].audit
    unresolved = replace(
        before,
        execution="unresolved",
        payload=replace(before.payload, verdict="inconclusive", witness=None),
    )
    assert ChangeResult("pass", "test", unresolved, passed).transition == "inconclusive"  # nosec B101
    assert (
        ChangeResult(
            "pass", "test", before, replace(passed, execution="unresolved")
        ).transition
        == "inconclusive"
    )  # nosec B101
    for execution in ("unsupported", "invalid"):
        candidate = rejected_result(
            before.contract,
            "audit",
            {"query": "audit"},
            "not supported",
            execution=execution,
        )
        assert (
            ChangeResult("unavailable", "test", before, candidate).transition
            == "inconclusive"
        )  # nosec B101
    partial = replace(
        before,
        execution="unresolved",
        payload=replace(
            before.payload, coverage=replace(before.payload.coverage, method="partial")
        ),
    )
    assert ChangeResult("pass", "test", partial, passed).transition == "restored"  # nosec B101


def test_distance_results_cannot_masquerade_as_audits(comparison):
    boundary = replace(comparison.baseline, request={"query": "boundary"})
    with pytest.raises(ValueError, match="requires an audit"):
        ChangeResult("wrong query", "test", comparison.baseline, boundary)


def test_fixed_to_adjustable_keeps_service_but_declares_new_permissions():
    model, loads = dispatch_example()
    model = replace(
        model,
        operating_limits=tuple(
            replace(r, limit=32) if r.name == "shared_fuel" else r
            for r in model.operating_limits
        ),
    )
    fixed = RecoursePolicy(
        "fixed",
        (
            DecisionRule("boiler_a", "MW", "operation", fixed_value=10),
            DecisionRule("boiler_b", "MW", "operation", fixed_value=6),
        ),
    )
    baseline = model.as_claim(loads, recourse=fixed)
    result = compare_changes(baseline, (change(model.as_claim(loads)),)).candidates[0]
    assert result.transition == "restored"  # nosec B101
    assert result.delta.changed_sections == ("operating",)  # nosec B101
    assert any("original policy" in note for note in result.delta.notes)  # nosec B101


@pytest.mark.parametrize(
    "field,value", [("limit", -2), ("tolerance", 0.01), ("residual_scale", 2)]
)
def test_selected_requirement_definition_changes_are_revisions(field, value):
    model, loads = example()
    changed = replace(
        model, requirements=(replace(model.requirements[0], **{field: value}),)
    )
    delta = compare_contracts(
        model.as_claim(loads).contract, changed.as_claim(loads).contract
    )
    assert delta.commitment == "revised" and not delta.requirements_preserved  # nosec B101


def test_physical_equation_change_is_not_equipment_restoration():
    model, loads = example()
    changed = replace(
        model, outputs=(replace(model.outputs[0], offset=2), model.outputs[1])
    )
    result = compare_changes(
        model.as_claim(loads), (change(changed.as_claim(loads)),)
    ).candidates[0]
    assert result.audit.payload.verdict == "pass"  # nosec B101
    assert result.delta.model_preserved is False  # nosec B101
    assert (
        result.delta.commitment == "unknown" and result.transition == "not_comparable"
    )  # nosec B101


def test_affine_limits_are_service_declarations_and_metric_is_separate():
    model, loads = affine_example()
    original = model.as_claim(loads)
    changed = replace(
        model,
        requirements=tuple(replace(r, limit=r.limit + 10) for r in model.requirements),
    )
    delta = compare_contracts(original.contract, changed.as_claim(loads).contract)
    assert delta.commitment == "revised"  # nosec B101
    distance = NormalizedLInf(loads.space)
    result = compare_changes(
        original, (change(model.as_claim(loads, distance=distance)),)
    ).candidates[0]
    assert result.delta.changed_sections == ("distance",)  # nosec B101
    assert result.delta.commitment == "preserved"  # nosec B101
    assert plain(result.distance) == distance.to_manifest()  # nosec B101
    assert result.audit.payload.search is None  # nosec B101


def test_unknown_adapter_does_not_guess_service_equivalence():
    model, loads = example()
    original = replace(model.as_claim(loads).contract, model_id="consumer/v1")
    same = compare_contracts(original, original)
    assert same.commitment == "preserved" and same.differences == ()  # nosec B101
    for changed in (
        replace(original, design={"capacity": 20}),
        replace(original, operating={**dict(original.operating), "extra": True}),
    ):
        delta = compare_contracts(original, changed)
        assert delta.commitment == "unknown" and delta.requirements_preserved is None  # nosec B101
    invalid_builtin = replace(original, model_id="linear_process/v1", model={})
    assert compare_contracts(original, invalid_builtin).commitment == "unknown"  # nosec B101


def test_equation_registry_change_cannot_hide_behind_builtin_model_id():
    model, loads = example()
    original = model.as_claim(loads).contract
    candidate = replace(
        original,
        constraints=tuple(
            replace(c, expression="changed physical balance")
            if c.kind == "equation"
            else c
            for c in original.constraints
        ),
    )
    delta = compare_contracts(original, candidate)
    assert delta.commitment == "unknown" and delta.model_preserved is False  # nosec B101
    assert "model" in delta.changed_sections  # nosec B101


def test_full_structural_difference_preserves_null_presence_and_pointer_escaping():
    model, loads = example()
    original = model.as_claim(loads).contract
    candidate = replace(original, design={"a/b~c": None})
    delta = compare_contracts(original, candidate)
    field = next(d for d in delta.differences if d.path.endswith("a~1b~0c"))
    assert not field.before_present and field.after_present  # nosec B101
    assert field.before is None and field.after is None  # nosec B101


def test_heat_equipment_service_and_physics_are_distinct():
    from operange.reference import example_claim

    baseline = example_claim()
    enlarged = replace(
        baseline,
        design=replace(baseline.design, source_capacity_mw=20, power_capacity_mw=20),
    )
    reduced_service = replace(
        baseline, requirement=replace(baseline.requirement, base_heat_mw=1)
    )
    result = baseline.as_claim().compare_changes(
        (
            change(enlarged.as_claim(), "equipment"),
            change(reduced_service.as_claim(), "service"),
        )
    )
    assert result.candidates[0].transition == "restored"  # nosec B101
    assert result.candidates[0].delta.changed_sections == ("design",)  # nosec B101
    assert result.candidates[1].audit.payload.verdict == "pass"  # nosec B101
    assert result.candidates[1].delta.commitment == "revised"  # nosec B101
    delta = compare_contracts(
        baseline.as_claim().contract, replace(baseline, cop=5).as_claim().contract
    )
    assert delta.commitment == "unknown" and delta.model_preserved is False  # nosec B101
    assert ChangeComparison.from_json(result.to_json(compact=True)).ref == result.ref  # nosec B101


def test_storage_equipment_and_earlier_information_keep_physical_futures():
    from operange.reference import example_storage_claim, StorageRepairAction

    baseline = example_storage_claim()
    equipment = replace(baseline, design=replace(baseline.design, capacity_mwh=2))
    earlier = baseline.with_repair(
        StorageRepairAction(
            "earlier signal",
            "Synthetic perfect advance information",
            reveal_before_preparation=True,
        )
    )
    result = baseline.as_claim().compare_changes(
        (
            change(equipment.as_claim(), "equipment"),
            change(earlier.as_claim(), "information"),
        )
    )
    assert result.baseline.payload.witness.kind == "incompatible_futures"  # nosec B101
    assert [c.transition for c in result.candidates] == ["restored", "restored"]  # nosec B101
    assert result.candidates[0].delta.changed_sections == ("design",)  # nosec B101
    assert result.candidates[1].delta.changed_sections == ("operating",)  # nosec B101
    assert all(c.delta.domain_preserved for c in result.candidates)  # nosec B101
    summary = next(r for r in result.candidates[1].requirements if r.relation == "eq")
    assert summary.unit == "MW" and summary.observed_margin == 0  # nosec B101
    assert summary.satisfied == 1  # nosec B101
    assert ChangeComparison.from_json(result.to_json(compact=True)).ref == result.ref  # nosec B101


def test_storage_duration_and_terminal_rules_are_service_changes():
    from operange.reference import example_storage_claim

    baseline = example_storage_claim()
    duration = replace(baseline, tree=replace(baseline.tree, event_hours=0.5))
    terminal = replace(
        baseline, requirement=replace(baseline.requirement, terminal_min_mwh=0.2)
    )
    for candidate in (duration, terminal):
        delta = compare_contracts(
            baseline.as_claim().contract, candidate.as_claim().contract
        )
        assert delta.requirements_preserved is False and delta.commitment == "revised"  # nosec B101


def test_startup_schedule_change_and_reduced_background_load_are_distinct():
    from examples.startup import example as startup_example

    baseline, staggered, domain = startup_example()
    result = baseline.as_claim(domain).compare_changes(
        (change(staggered.as_claim(domain)),)
    )
    candidate = result.candidates[0]
    assert candidate.transition == "restored"  # nosec B101
    assert candidate.delta.changed_sections == ("operating",)  # nosec B101
    summary = next(
        r for r in result.requirements if r.constraint_ref == "shared_capacity"
    )
    assert summary.unit == "kVA" and summary.observed_margin == pytest.approx(-200)  # nosec B101
    reduced = replace(baseline, background_load=0)
    delta = compare_contracts(
        baseline.as_claim(domain).contract, reduced.as_claim(domain).contract
    )
    assert delta.commitment == "revised" and not delta.requirements_preserved  # nosec B101
    assert ChangeComparison.from_json(result.to_json(compact=True)).ref == result.ref  # nosec B101


def test_pinch_utility_change_retains_target_scope():
    from examples.pinch import example as pinch_example

    model, loads = pinch_example()
    enlarged = replace(model, hot_utility_capacity_kw=100)
    result = model.as_claim(loads).compare_changes((change(enlarged.as_claim(loads)),))
    candidate = result.candidates[0]
    assert candidate.transition == "restored"  # nosec B101
    assert candidate.delta.changed_sections == ("design",)  # nosec B101
    assert (
        candidate.audit.contract.model["installed_network_feasibility"]
        == "not_assessed"
    )  # nosec B101
    assert ChangeComparison.from_json(result.to_json(compact=True)).ref == result.ref  # nosec B101
