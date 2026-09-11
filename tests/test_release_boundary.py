"""Release-boundary semantics and pre-refactor portable result compatibility."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

import operange as process
from operange.reference import (
    FrozenHeatRecoveryContract,
    HeatRecoveryDesign,
    RepairAction,
    StorageDesign,
    StorageRepairAction,
    example_claim,
    example_storage_claim,
)


def test_reference_changes_preserve_order_without_valuation_or_selection():
    heat = example_claim()
    changes = (
        RepairAction("Z larger", HeatRecoveryDesign(8, 2), "Test"),
        RepairAction("A baseline", heat.design, "Test"),
    )
    comparison = heat.compare_repairs(changes)
    assert tuple(r.action for r in comparison.candidates) == changes  # nosec B101
    assert [r.audit.status for r in comparison.candidates] == ["pass", "fail"]  # nosec B101
    assert not hasattr(comparison, "selected")  # nosec B101
    assert not hasattr(comparison, "to_markdown")  # nosec B101
    assert set(asdict(changes[0])) == {"name", "design", "provenance"}  # nosec B101
    changed = heat.with_repair(changes[0])
    assert changed.uncertainty == heat.uncertainty  # nosec B101
    assert changed.requirement == heat.requirement  # nosec B101
    with pytest.raises(ValueError):
        heat.with_repair(object())

    storage = example_storage_claim()
    actions = (
        StorageRepairAction("Z larger", "Test", design=StorageDesign(capacity_mwh=2)),
        StorageRepairAction(
            "A earlier", "Test exact signal", reveal_before_preparation=True
        ),
    )
    comparison = storage.compare_repairs(actions)
    assert tuple(r.action for r in comparison.candidates) == actions  # nosec B101
    assert not hasattr(comparison, "selected")  # nosec B101
    assert all(r.causal.status == "pass" for r in comparison.candidates)  # nosec B101
    assert storage.with_repair(actions[0]).tree == storage.tree  # nosec B101
    assert storage.with_repair(actions[1]).tree.domain == storage.tree.domain  # nosec B101
    assert heat.compare_repairs(()).candidates == ()  # nosec B101
    assert storage.compare_repairs(()).candidates == ()  # nosec B101


def test_shared_exports_are_distinct_from_reference_and_experimental_apis():
    for name in (
        "RepairAction",
        "StorageRepairAction",
        "HeatRecoveryClaim",
        "ScenarioTree",
        "QuadraticProcessAdapter",
        "SensibleHeatStream",
        "HeatCascadeAdapter",
    ):
        assert name not in process.__all__  # nosec B101
    assert callable(process.bind_contract) and callable(process.rejected_result)  # nosec B101


def test_pre_refactor_v1_results_and_frozen_contract_still_load():
    corpus = json.loads(
        (Path(__file__).parent / "fixtures/prototype_result_v1.json").read_text()
    )
    for name in ("evaluation", "sensitivity", "robustness"):
        restored = process.result_from_json(json.dumps(corpus[name]))
        assert restored.to_dict() == corpus[name]  # nosec B101
    frozen = FrozenHeatRecoveryContract.from_json(json.dumps(corpus["frozen_heat"]))
    assert json.loads(frozen.to_json()) == corpus["frozen_heat"]  # nosec B101


def test_engineering_repair_metadata_rejects_invalid_provenance():
    action = RepairAction("change", HeatRecoveryDesign(), "Test")
    with pytest.raises(ValueError):
        replace(action, provenance="")
