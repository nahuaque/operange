"""Freeze a causal storage rule and replay new paths using the installed API."""

import json

from operange import EngineeringChange, FrozenController
from operange.reference import (
    ScenarioTree,
    StorageController,
    StorageDesign,
    StoragePreparationRule,
    StorageScenario,
    ThermalStorageClaim,
)


def example():
    return ThermalStorageClaim(
        "Meet demand and absorb recovered heat with a 2 MWh store",
        StorageDesign(capacity_mwh=2, discharging_power_mw=2),
        ScenarioTree(
            (
                StorageScenario(
                    "Demand surge", 1, 0, "Synthetic one-hour demand event"
                ),
                StorageScenario(
                    "Heat surplus", 0, 1, "Synthetic mandatory heat recovery"
                ),
            )
        ),
    )


def preparation(charge_mw):
    return StorageController(
        f"Prepare at {charge_mw:g} MW",
        (StoragePreparationRule("unrevealed", charge_mw),),
        "Synthetic prescribed preparation; no fitting or future-event access",
    )


def run_example():
    model = example()
    baseline = model.as_claim(controller=preparation(0.5))
    revised = model.as_claim(controller=preparation(1))
    comparison = baseline.compare_changes(
        (
            EngineeringChange(
                "Prepare 1 MWh", revised, "Revise the saved preparation command"
            ),
            EngineeringChange(
                "Adjustable causal benchmark",
                model.as_claim(),
                "Separate policy-class audit",
            ),
        )
    )
    frozen = revised.freeze()
    loaded = FrozenController.from_json(frozen.to_json())
    held_out = ScenarioTree(
        (
            StorageScenario("Later demand", 0.75, 0, "Synthetic held-out path"),
            StorageScenario("Later recovery", 0, 0.75, "Synthetic held-out path"),
            StorageScenario(
                "Demand beyond reserve", 1.25, 0, "Synthetic held-out failure"
            ),
        )
    )
    unseen = ScenarioTree(
        (StorageScenario("New signal", 0.75, 0, "Unlisted observation", "new signal"),)
    )
    return {
        "comparison": comparison.to_dict(compact=True),
        "frozen_controller": frozen.to_dict(),
        "replayed_audit": loaded.audit_result().to_dict(compact=True),
        "held_out_audit": loaded.as_claim(tree=held_out)
        .audit_result()
        .to_dict(compact=True),
        "unknown_signal_audit": loaded.as_claim(tree=unseen)
        .audit_result()
        .to_dict(compact=True),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
