"""Independent energy-balance and information-timing checks for storage."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from operange.reference import (
    ScenarioTree,
    StorageDesign,
    StorageRepairAction,
    StorageRequirement,
    StorageScenario,
    ThermalStorageClaim,
)
from operange import linear
from operange.reference import example_storage_claim


def only(claim, name):
    return replace(
        claim,
        tree=replace(
            claim.tree,
            scenarios=tuple(s for s in claim.tree.scenarios if s.name == name),
        ),
    )


def test_individually_recoverable_futures_require_incompatible_shared_actions():
    claim = example_storage_claim()
    fixed = claim.audit("fixed", fixed_preparation_power_mw=0.5)
    causal = claim.audit()
    foresight = claim.audit("perfect_foresight")
    assert [fixed.status, causal.status, foresight.status] == ["fail", "fail", "pass"]  # nosec B101
    assert fixed.witness.kind == "fixed_policy_failure"  # nosec B101
    assert len(fixed.witness.scenarios) == 1  # nosec B101
    assert causal.witness.kind == "conflicting_futures"  # nosec B101
    assert causal.witness.scenarios == ("Demand surge", "Heat surplus")  # nosec B101
    assert causal.witness.minimum_prepared_energy_mwh == 1  # nosec B101
    assert causal.witness.maximum_prepared_energy_mwh == 0  # nosec B101
    assert causal.witness.gap_mwh == 1  # nosec B101
    assert causal.witness.minimum_cardinality  # nosec B101
    for scenario in causal.witness.scenarios:
        assert only(claim, scenario).audit().status == "pass"  # nosec B101
    plans = {row.scenario: row for row in foresight.dispatch}
    assert plans["Demand surge"].preparation_charge_mw == pytest.approx(1)  # nosec B101
    assert plans["Heat surplus"].preparation_charge_mw == pytest.approx(0)  # nosec B101
    assert plans["Demand surge"].terminal_energy_mwh == pytest.approx(0)  # nosec B101
    assert plans["Heat surplus"].terminal_energy_mwh == pytest.approx(1)  # nosec B101
    assert foresight.evidence.reference_error <= claim.tolerance  # nosec B101


def test_extra_future_and_input_order_do_not_change_minimal_pair():
    claim = example_storage_claim()
    neutral = StorageScenario("Neutral", 0, 0, "test")
    tree = replace(claim.tree, scenarios=(*reversed(claim.tree.scenarios), neutral))
    audit = replace(claim, tree=tree).audit()
    assert audit.witness.scenarios == ("Demand surge", "Heat surplus")  # nosec B101
    assert len(audit.intervals) == 3  # nosec B101
    assert replace(claim, tree=tree).audit("perfect_foresight").status == "pass"  # nosec B101


def test_only_observable_history_controls_preparation_sharing():
    claim = example_storage_claim()
    revealed = replace(
        claim.tree,
        scenarios=tuple(
            replace(s, preparation_observation=s.name) for s in claim.tree.scenarios
        ),
    )
    known = replace(claim, tree=revealed)
    assert known.audit().status == "pass"  # nosec B101
    assert known.audit("fixed", fixed_preparation_power_mw=0.5).status == "fail"  # nosec B101
    # Reusing a signal label hides the distinction again, regardless of names.
    hidden = replace(
        revealed,
        scenarios=tuple(
            replace(s, preparation_observation="same signal")
            for s in revealed.scenarios
        ),
    )
    assert replace(claim, tree=hidden).audit().status == "fail"  # nosec B101
    assert replace(claim, tree=hidden).audit("perfect_foresight").status == "pass"  # nosec B101


def test_partial_information_groups_only_indistinguishable_futures():
    claim = example_storage_claim()
    demand, surplus = claim.tree.scenarios
    tree = replace(
        claim.tree,
        scenarios=(
            replace(demand, preparation_observation="demand family"),
            StorageScenario("Small demand", 0.5, 0, "test", "demand family"),
            replace(surplus, preparation_observation="surplus family"),
        ),
    )
    audit = replace(claim, tree=tree).audit()
    assert audit.status == "pass"  # nosec B101
    plans = {row.scenario: row for row in audit.dispatch}
    assert plans["Demand surge"].preparation_charge_mw == pytest.approx(1)  # nosec B101
    assert plans["Small demand"].preparation_charge_mw == pytest.approx(1)  # nosec B101
    assert plans["Heat surplus"].preparation_charge_mw == pytest.approx(0)  # nosec B101
    assert plans["Small demand"].terminal_energy_mwh == pytest.approx(0.5)  # nosec B101


def test_larger_store_admits_one_shared_preparation_action():
    claim = replace(example_storage_claim(), design=StorageDesign(capacity_mwh=2))
    audit = claim.audit()
    assert audit.status == "pass"  # nosec B101
    assert {row.preparation_information for row in audit.dispatch} == {"unrevealed"}  # nosec B101
    for row in audit.dispatch:
        assert row.preparation_charge_mw == pytest.approx(1)  # nosec B101
        assert row.prepared_energy_mwh == pytest.approx(1)  # nosec B101
    assert sorted(row.terminal_energy_mwh for row in audit.dispatch) == pytest.approx(
        [0, 2]
    )  # nosec B101
    assert claim.audit("fixed", fixed_preparation_power_mw=1).status == "pass"  # nosec B101
    assert claim.audit("fixed", fixed_preparation_power_mw=0.5).status == "fail"  # nosec B101


def test_period_lengths_losses_efficiencies_and_terminal_reserve_are_obeyed():
    claim = ThermalStorageClaim(
        "Explicit energy accounting",
        StorageDesign(
            capacity_mwh=4,
            charging_power_mw=2,
            discharging_power_mw=2,
            charge_efficiency=0.8,
            discharge_efficiency=0.5,
            loss_rate_per_hour=0.1,
        ),
        ScenarioTree(
            (
                StorageScenario("Demand", 0.5, 0, "test"),
                StorageScenario("Surplus", 0, 0.25, "test"),
            ),
            preparation_hours=2,
            event_hours=2,
        ),
        StorageRequirement(
            initial_energy_mwh=0.5, terminal_min_mwh=0.2, terminal_max_mwh=3
        ),
    )
    audit = claim.audit()
    assert audit.status == "pass"  # nosec B101
    for row in audit.dispatch:
        assert row.prepared_energy_mwh == pytest.approx(2.75)  # nosec B101
        assert row.preparation_charge_mw == pytest.approx(1.46875)  # nosec B101
        initial_retained = 0.5 * (1 - 0.1 * 2)
        assert row.prepared_energy_mwh == pytest.approx(
            initial_retained + 0.8 * row.preparation_charge_mw * 2
        )  # nosec B101
        final = (1 - 0.1 * 2) * row.prepared_energy_mwh + (
            0.8 * row.event_charge_mw - row.event_discharge_mw / 0.5
        ) * 2
        assert row.terminal_energy_mwh == pytest.approx(final)  # nosec B101
        assert 0.2 - claim.tolerance <= final <= 3 + claim.tolerance  # nosec B101
        scenario = next(s for s in claim.tree.scenarios if s.name == row.scenario)
        assert scenario.recovered_heat_mw + row.event_discharge_mw == pytest.approx(
            scenario.demand_mw + row.event_charge_mw
        )  # nosec B101
        assert row.event_charge_mw * row.event_discharge_mw == 0  # nosec B101
    assert sorted(row.terminal_energy_mwh for row in audit.dispatch) == pytest.approx(
        [0.2, 2.6]
    )  # nosec B101


def test_equipment_upgrade_cannot_relax_explicit_terminal_commitment():
    claim = replace(
        example_storage_claim(), requirement=StorageRequirement(terminal_max_mwh=1)
    )
    enlarged = claim.with_repair(example_storage_repairs()[-1])
    assert enlarged.design.capacity_mwh == 2  # nosec B101
    assert enlarged.requirement == claim.requirement  # nosec B101
    assert enlarged.audit().status == "fail"  # nosec B101
    assert enlarged.audit().witness.kind == "conflicting_futures"  # nosec B101


def test_impossible_terminal_reserve_is_a_claim_failure():
    claim = replace(
        example_storage_claim(), requirement=StorageRequirement(terminal_min_mwh=2)
    )
    assert claim.audit().status == "fail"  # nosec B101
    assert claim.audit("perfect_foresight").status == "fail"  # nosec B101
    assert claim.audit().witness.kind == "individual_infeasibility"  # nosec B101


@pytest.mark.parametrize(
    "scenario,design,reason",
    [
        ("Demand surge", StorageDesign(discharging_power_mw=0.5), "discharging"),
        ("Heat surplus", StorageDesign(charging_power_mw=0.5), "charging"),
    ],
)
def test_event_power_limits_are_separate_from_energy_limits(scenario, design, reason):
    claim = replace(only(example_storage_claim(), scenario), design=design)
    audit = claim.audit("perfect_foresight")
    assert audit.status == "fail"  # nosec B101
    assert audit.witness.kind == "individual_infeasibility"  # nosec B101
    assert reason in audit.witness.explanation  # nosec B101


def test_model_cannot_dump_surplus_cycle_losses_or_reset_initial_energy():
    claim = only(example_storage_claim(), "Heat surplus")
    claim = replace(
        claim,
        requirement=StorageRequirement(initial_energy_mwh=1),
        design=StorageDesign(charge_efficiency=0.5, discharge_efficiency=0.5),
    )
    # Simultaneous charge/discharge could dissipate heat; it is not allowed.
    assert claim.audit("perfect_foresight").status == "fail"  # nosec B101
    assert claim.audit().witness.kind == "individual_infeasibility"  # nosec B101


def test_preparation_duration_changes_reachability_without_future_leakage():
    claim = example_storage_claim()
    short = replace(claim, tree=replace(claim.tree, preparation_hours=0.5))
    assert only(short, "Demand surge").audit("perfect_foresight").status == "fail"  # nosec B101
    long = replace(claim, tree=replace(claim.tree, preparation_hours=2))
    assert long.audit("perfect_foresight").status == "pass"  # nosec B101
    assert long.audit().status == "fail"  # nosec B101


@pytest.mark.parametrize(
    "changes",
    [
        {"capacity_mwh": -1},
        {"charging_power_mw": float("nan")},
        {"discharging_power_mw": float("inf")},
        {"charge_efficiency": 0},
        {"discharge_efficiency": 1.01},
        {"loss_rate_per_hour": -1},
        {"capacity_mwh": True},
    ],
)
def test_invalid_designs_rejected(changes):
    with pytest.raises(ValueError):
        StorageDesign(**changes)


def test_invalid_trees_and_time_contracts_rejected():
    claim = example_storage_claim()
    for hours in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(claim.tree, event_hours=hours)
    with pytest.raises(ValueError):
        ScenarioTree(())
    with pytest.raises(ValueError):
        ScenarioTree((claim.tree.scenarios[0],) * 2)
    with pytest.raises(ValueError):
        replace(claim.tree.scenarios[0], preparation_observation="")
    with pytest.raises(ValueError):
        replace(claim.tree.scenarios[0], demand_mw=-1)
    with pytest.raises(ValueError):
        replace(claim, design=StorageDesign(loss_rate_per_hour=1))
    with pytest.raises(ValueError):
        replace(claim, requirement=StorageRequirement(initial_energy_mwh=2))
    with pytest.raises(ValueError):
        StorageRequirement(terminal_min_mwh=2, terminal_max_mwh=1)
    with pytest.raises(ValueError):
        StorageRepairAction("bad", "test", reveal_before_preparation=1)
    with pytest.raises(ValueError):
        claim.audit("unknown")
    with pytest.raises(ValueError):
        claim.audit("causal", fixed_preparation_power_mw=0.5)
    for value in (None, -1, 2, float("nan"), True):
        with pytest.raises(ValueError):
            claim.audit("fixed", fixed_preparation_power_mw=value)


def test_false_solver_infeasibility_is_not_a_physical_certificate(monkeypatch):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False, status=2, message="infeasible", x=None
        ),
    )
    assert example_storage_claim().audit("perfect_foresight").status == "inconclusive"  # nosec B101


@pytest.mark.parametrize("x", [[0, 0, 0, 0, 0], [1, 1, 0, 0.5, 0.5]])
def test_success_flag_does_not_bypass_balances_or_independent_optimum(monkeypatch, x):
    claim = only(example_storage_claim(), "Demand surge")
    claim = replace(
        claim,
        design=StorageDesign(capacity_mwh=2, charging_power_mw=2),
        tree=replace(
            claim.tree, scenarios=(replace(claim.tree.scenarios[0], demand_mw=0.5),)
        ),
    )
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(success=True, status=0, message="optimal", x=x),
    )
    audit = claim.audit()
    assert audit.status == "inconclusive"  # nosec B101
    assert audit.dispatch == ()  # nosec B101


def test_shared_lp_adapter_validates_shape_and_finite_coefficients():
    for options in (
        {"equalities": [[1, 2]], "rhs": [1]},
        {"equalities": [[1]], "rhs": []},
        {"inequalities": [[float("nan")]], "upper": [1]},
    ):
        with pytest.raises(ValueError):
            linear.solve_lp([1], [(0, 1)], 1e-8, **options)
    with pytest.raises(ValueError):
        linear.solve_lp([1], [(1, 0)], 1e-8)


def example_storage_repairs():
    return (
        StorageRepairAction("Baseline", "Test equipment change"),
        StorageRepairAction(
            "More charging power",
            "Test equipment change",
            design=StorageDesign(charging_power_mw=2),
        ),
        StorageRepairAction(
            "Earlier event notification",
            "Test exact signal",
            reveal_before_preparation=True,
        ),
        StorageRepairAction(
            "Larger store",
            "Test equipment change",
            design=StorageDesign(capacity_mwh=2),
        ),
    )
