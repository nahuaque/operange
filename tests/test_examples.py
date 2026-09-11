"""Pricing and presentation are consumer behavior, outside the process wheel."""

from dataclasses import replace
import json
from types import SimpleNamespace
import pytest

from operange import result_from_json, linear
from operange.reference import HeatDemand
from examples.heat_recovery import example_claim, example_repairs, run_example
from examples.thermal_storage import (
    example_storage_claim,
    example_storage_repairs,
    run_storage_example,
)
from examples.reporting import (
    PricedRepair,
    compare_heat_repairs,
    compare_storage_repairs,
)


def test_consumer_price_validation_is_independent_of_engineering_action():
    change = example_repairs()[0].change
    for price in (-1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            PricedRepair(change, price, "Synthetic cost estimate")
    with pytest.raises(ValueError):
        PricedRepair(change, 0, "")
    free = PricedRepair(change, 0, "No incremental installed cost")
    assert free.change == change  # nosec B101


def test_demo_exports_are_consumable_and_contain_no_business_fields():
    for payload in (run_example()[1], run_storage_example()[1]):
        for data in payload["engineering_results"].values():
            result = result_from_json(json.dumps(data))
            assert result.schema_version == "process_result/v1"  # nosec B101
            text = result.to_json().lower()
            assert all(
                s not in text
                for s in (
                    "capex",
                    "gbp",
                    "selected_repair",
                    "added_cost",
                )
            )  # nosec B101


def test_repairs_retain_futures_and_explicitly_change_information():
    claim = example_storage_claim()
    actions = example_storage_repairs()
    comparison = compare_storage_repairs(claim, tuple(reversed(actions)))
    assert [r.causal.status for r in comparison.candidates] == [
        "fail",
        "fail",
        "pass",
        "pass",
    ]  # nosec B101
    assert all(r.perfect_foresight.status == "pass" for r in comparison.candidates)  # nosec B101
    assert comparison.selected.action.change.name == "Earlier event notification"  # nosec B101
    assert comparison.selected.action.added_capex_gbp == 15_000  # nosec B101
    equipment_only = compare_storage_repairs(
        claim, [a for a in actions if not a.change.reveal_before_preparation]
    )
    assert equipment_only.selected.action.change.name == "Larger store"  # nosec B101
    changed = claim.with_repair(comparison.selected.action.change)
    assert changed.requirement == claim.requirement  # nosec B101
    assert changed.design == claim.design  # nosec B101
    assert changed.tree.preparation_hours == claim.tree.preparation_hours  # nosec B101
    for before, after in zip(claim.tree.scenarios, changed.tree.scenarios):
        assert (
            before.name,
            before.demand_mw,
            before.recovered_heat_mw,
            before.provenance,
        ) == (after.name, after.demand_mw, after.recovered_heat_mw, after.provenance)  # nosec B101
        assert before.preparation_observation != after.preparation_observation  # nosec B101
    assert compare_storage_repairs(claim, ()).selected is None  # nosec B101
    with pytest.raises(ValueError, match="unique"):
        compare_storage_repairs(claim, (actions[0], actions[0]))


@pytest.mark.parametrize("solver_status", [1, 3, 4])
def test_unresolved_solver_does_not_certify_a_conflicting_tree(
    monkeypatch, solver_status
):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False, status=solver_status, message="injected failure", x=None
        ),
    )
    claim = example_storage_claim()
    for mode in ("causal", "perfect_foresight"):
        audit = claim.audit(mode)
        assert audit.status == "inconclusive"  # nosec B101
        assert audit.witness is None  # nosec B101
        assert audit.dispatch == ()  # nosec B101
    assert compare_storage_repairs(claim, example_storage_repairs()).selected is None  # nosec B101
    report, payload = run_storage_example()
    assert payload["selected_repair"] is None  # nosec B101
    assert "comparison was not fully verified" in report  # nosec B101
    assert "benchmark would accept" not in report  # nosec B101


def test_review_artifact_carries_tree_control_scope_and_complete_decisions():
    report, payload = run_storage_example()
    assert payload["causal"]["witness"]["scenarios"] == ("Demand surge", "Heat surplus")  # nosec B101
    assert payload["perfect_foresight"]["status"] == "pass"  # nosec B101
    assert payload["selected_repair"] == "Earlier event notification"  # nosec B101
    assert "no shared preparation decision" in report  # nosec B101
    assert "£15,000" in report  # nosec B101
    assert "continuous-domain robustness radius" in report  # nosec B101
    assert payload["causal"]["contract"]["requirement"]["initial_energy_mwh"] == 0  # nosec B101
    assert all(
        s["preparation_observation"] == "unrevealed"
        for s in payload["causal"]["contract"]["tree"]["scenarios"]
    )  # nosec B101
    json.dumps(payload, allow_nan=False)


def test_repairs_preserve_original_claim_and_find_cheapest_passing_catalog_entry():
    claim = example_claim()
    comparison = compare_heat_repairs(claim, tuple(reversed(example_repairs())))
    assert [row.audit.status for row in comparison.candidates] == [  # nosec B101
        "fail",
        "fail",
        "fail",
        "pass",
        "pass",
    ]
    assert [row.audit.worst.slack_mw for row in comparison.candidates] == pytest.approx(  # nosec B101
        [-1.5, -1.5, -0.5, 0, 0.5]
    )
    assert comparison.selected.action.added_capex_gbp == 240_000  # nosec B101
    assert comparison.selected.action.change.name == "Source-capture upgrade"  # nosec B101
    modified = claim.with_repair(comparison.selected.action.change)
    assert modified.requirement == claim.requirement  # nosec B101
    assert modified.uncertainty == claim.uncertainty  # nosec B101
    assert modified.recourse == claim.recourse  # nosec B101
    assert claim.design.source_capacity_mw == 6  # nosec B101
    reserve = replace(claim, requirement=HeatDemand(6, reserve_mw=0.1))
    assert (  # nosec B101
        compare_heat_repairs(reserve, example_repairs()).selected.action.added_capex_gbp
        == 310_000
    )
    assert replace(claim, requirement=HeatDemand(4.8)).audit().status == "pass"  # nosec B101
    assert compare_heat_repairs(claim, ()).selected is None  # nosec B101
    assert compare_heat_repairs(claim, example_repairs()[:3]).selected is None  # nosec B101
    with pytest.raises(ValueError, match="unique"):
        compare_heat_repairs(claim, (example_repairs()[0], example_repairs()[0]))


def test_demo_produces_complete_reviewable_artifacts():
    markdown, payload, policy = run_example()
    assert "0.571429" in markdown  # nosec B101
    assert "0.574286" in markdown  # nosec B101
    assert "zero worst-case headroom" in markdown  # nosec B101
    assert "revised claim" in markdown  # nosec B101
    assert "£240,000" in markdown  # nosec B101
    assert "not a probability" in markdown  # nosec B101
    assert payload["witness"]["minimum_verified"]  # nosec B101
    assert payload["replay"]["status"] == "pass"  # nosec B101
    assert policy.fingerprint == payload["selected_contract_fingerprint"]  # nosec B101
    json.dumps(payload, allow_nan=False)
