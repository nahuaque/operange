"""Compact transport preserves v1 identities, validation and saved artifacts."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from operange import (
    FiniteSet,
    PiecewiseLinearProfile,
    Scenario,
    result_from_json,
)
from operange.reference import (
    StartupEvent,
    StartupLoadAdapter,
    example_claim,
    example_storage_claim,
)


def startup_audit(count):
    profile = PiecewiseLinearProfile((0, 1), (1, 0.5), "kW", "test")
    event = StartupEvent("motor", profile, 0, "test")
    model = StartupLoadAdapter("test", (event,), 2, 2)
    domain = FiniteSet(
        model.input_space,
        tuple(
            Scenario(
                str(i),
                {
                    event.input_name("amplitude_scale"): 1 + i / 10000,
                    event.input_name("duration_scale"): 1,
                    event.input_name("timing_jitter_seconds"): 0,
                },
                "test",
            )
            for i in range(count)
        ),
    )
    return model.as_claim(domain).audit_result()


def test_compact_round_trip_all_result_families_and_historical_v1_without_execution(
    monkeypatch,
):
    heat = example_claim()
    results = [
        heat.evaluate_result(heat.uncertainty.nominal),
        heat.sensitivity_result({"source_derating": 0.1, "demand_increase": 0.05}),
        heat.audit_result(),
        heat.breaking_result(),
        heat.evaluate_result({}),
        example_storage_claim().audit_result(),
        startup_audit(5),
    ]
    corpus = json.loads(
        (Path(__file__).parent / "fixtures/prototype_result_v1.json").read_text()
    )
    results.extend(
        result_from_json(json.dumps(corpus[k]))
        for k in ("evaluation", "sensitivity", "robustness")
    )

    def no_solve(*args, **kwargs):
        raise AssertionError("loading may not execute a model")

    monkeypatch.setattr("operange.linear.linprog", no_solve)
    for result in results:
        legacy = result.to_dict()
        compact = result.to_dict(compact=True)
        assert compact["schema_version"] == "process_result_bundle/v1"  # nosec B101
        assert len(compact["contracts"]) == 1  # nosec B101
        restored = result_from_json(result.to_json(compact=True))
        assert restored == result and restored.to_dict() == legacy  # nosec B101
        assert type(result).from_json(json.dumps(compact)).result_id == result.result_id  # nosec B101
        if restored.supporting_evaluations:
            assert all(
                e.contract is restored.contract for e in restored.supporting_evaluations
            )  # nosec B101


@pytest.mark.parametrize(
    "corruption",
    [
        "contract_content",
        "evaluation_content",
        "root_content",
        "missing_contract",
        "missing_evaluation",
        "cycle",
        "unused_contract",
        "unused_evaluation",
        "duplicate_evaluation",
        "wrong_kind",
        "unknown_field",
        "null_evaluation",
    ],
)
def test_compact_loader_rejects_tampering_and_broken_references(corruption):
    data = example_claim().audit_result().to_dict(compact=True)
    root = data["result"]
    contract_id = next(iter(data["contracts"]))
    evaluation_id = next(iter(data["evaluations"]))
    evaluation = data["evaluations"][evaluation_id]
    if corruption == "contract_content":
        data["contracts"][contract_id]["design"]["power_capacity_mw"] = 999
    elif corruption == "evaluation_content":
        evaluation["request"]["realization"]["source_derating"] = 0.123
    elif corruption == "root_content":
        root["request"]["query"] = "different"
    elif corruption == "missing_contract":
        data["contracts"].clear()
    elif corruption == "missing_evaluation":
        data["evaluations"].clear()
    elif corruption == "cycle":
        evaluation["artifacts"]["evaluations"] = [evaluation_id]
    elif corruption == "unused_contract":
        data["contracts"]["unused"] = data["contracts"][contract_id]
    elif corruption == "unused_evaluation":
        data["evaluations"]["unused"] = evaluation
    elif corruption == "duplicate_evaluation":
        root["artifacts"]["evaluations"].append(evaluation_id)
    elif corruption == "wrong_kind":
        evaluation["kind"] = "robustness"
    elif corruption == "unknown_field":
        data["unexpected"] = True
    else:
        data["evaluations"][evaluation_id] = None
    with pytest.raises(ValueError):
        result_from_json(json.dumps(data))


def test_compact_size_tracks_scenarios_without_repeating_the_domain():
    small, large = startup_audit(10), startup_audit(40)
    small_size = len(small.to_json(compact=True, indent=None))
    large_size = len(large.to_json(compact=True, indent=None))
    assert large_size < 5 * small_size  # nosec B101
    assert large_size < len(large.to_json(indent=None)) / 2  # nosec B101


def test_cached_identities_do_not_leak_through_exports_or_dataclass_replacement():
    result = example_claim().evaluate_result(
        {"source_derating": 0.1, "demand_increase": 0.05}
    )
    original_id = result.result_id
    exported = result.to_dict()
    exported["request"]["query"] = "mutated"
    assert result.result_id == original_id  # nosec B101
    changed = replace(result, request={**result.request, "note": "different request"})
    assert changed.result_id != original_id  # nosec B101
    contract = replace(
        result.contract, model={**result.contract.model, "note": "changed"}
    )
    assert contract.ref != result.contract_ref  # nosec B101
    domain = example_claim().uncertainty
    before = domain.ref
    changed_domain = replace(
        domain,
        parameters=tuple(replace(p, provenance="updated") for p in domain.parameters),
    )
    assert changed_domain.ref != before  # nosec B101
