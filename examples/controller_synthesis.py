"""Synthesize, independently audit, freeze and replay an affine boiler controller.

Run with ``uv run --extra cvxpy python -m examples.controller_synthesis``.
"""

from dataclasses import replace
import json

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    ControlTarget,
    ControlTrackingObjective,
    FrozenController,
    LinearControl,
    LinearObjective,
    LinearProcessAdapter,
    Parameter,
)


def example():
    domain = BoxSet(
        tuple(
            Parameter(n, "MW", nominal, low, high, 2, "Synthetic steam demand")
            for n, nominal, low, high in (("dryer", 10, 8, 12), ("evaporator", 6, 4, 8))
        )
    )
    model = LinearProcessAdapter(
        "Synthetic boilers with a 32 MW fuel supply",
        domain.space,
        (
            AffineOutput(
                "steam_balance",
                "MW",
                "thermal_power",
                0,
                tuple(
                    AffineTerm(n, a, "MW/MW")
                    for n, a in (
                        ("boiler_a", 1),
                        ("boiler_b", 1),
                        ("dryer", -1),
                        ("evaporator", -1),
                    )
                ),
                "Delivered steam minus demand",
            ),
            AffineOutput(
                "fuel",
                "MW",
                "thermal_power",
                0,
                (
                    AffineTerm("boiler_a", 1.25, "MW/MW"),
                    AffineTerm("boiler_b", 2, "MW/MW"),
                ),
                "Fixed boiler efficiencies of 80% and 50%",
            ),
        ),
        (AffineRequirement("meet_load", "steam_balance", 0, "ge"),),
        (
            LinearControl("boiler_a", "MW", 0, 12),
            LinearControl("boiler_b", "MW", 0, 10),
        ),
        (
            AffineRequirement("no_excess_steam", "steam_balance", 0),
            AffineRequirement("shared_fuel", "fuel", 32),
        ),
        objective=LinearObjective("fuel"),
    )
    return model.as_claim(domain)


def run_example():
    claim = example()
    fuel = claim.synthesize_controller(name="Minimum worst-case fuel")
    tracked = claim.synthesize_controller(
        name="Preferred boiler commands",
        objective=ControlTrackingObjective(
            (
                ControlTarget("boiler_a", 10, 1, "MW"),
                ControlTarget("boiler_b", 6, 1, "MW"),
            )
        ),
    )
    result = {
        "fuel": fuel.to_dict(compact=True),
        "tracking": tracked.to_dict(compact=True),
    }
    if fuel.verified:
        frozen = fuel.freeze()
        loaded = FrozenController.from_json(frozen.to_json())
        result.update(
            frozen_controller=frozen.to_dict(),
            replayed_audit=loaded.audit_result().to_dict(compact=True),
            interior=loaded.evaluate_result({"dryer": 9.75, "evaporator": 5.5}).to_dict(
                compact=True
            ),
        )
    # Removing all observations restricts both controllers to constant commands.
    # Failure to find such a controller says nothing about adjustable feasibility.
    constant_permissions = replace(
        claim.recourse,
        rules=tuple(replace(r, observes=()) for r in claim.recourse.rules),
    )
    constant = replace(claim, recourse=constant_permissions).synthesize_controller()
    result["constant_policy_search"] = constant.to_dict(compact=True)
    return result


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
