"""Guaranteed production with equal boiler allocation or adjustable dispatch."""

import json

from operange import (
    AffineControlRule,
    AffineController,
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    LinearControl,
    LinearProcessAdapter,
    OperatingLimitResult,
    Parameter,
    verify_result,
)


def example():
    domain = BoxSet(
        (
            Parameter(
                "production", "t/h", 8, 0, 12, 1, "Declared production search range"
            ),
            Parameter("auxiliary", "MW", 1.5, 1, 2, 0.5, "Independent utility demand"),
        )
    )
    model = LinearProcessAdapter(
        "Two boilers serving production and auxiliary demand",
        domain.space,
        outputs=(
            AffineOutput(
                "steam_balance",
                "MW",
                "thermal_power",
                0,
                (
                    AffineTerm("boiler_a", 1, "MW/MW"),
                    AffineTerm("boiler_b", 1, "MW/MW"),
                    AffineTerm("production", -2, "MW/t/h"),
                    AffineTerm("auxiliary", -1, "MW/MW"),
                ),
                "Two MW of steam per tonne/hour, plus auxiliary demand",
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
        requirements=(AffineRequirement("serve_load", "steam_balance", 0, "ge"),),
        controls=(
            LinearControl("boiler_a", "MW", 0, 12),
            LinearControl("boiler_b", "MW", 0, 10),
        ),
        operating_limits=(
            AffineRequirement("no_excess", "steam_balance", 0),
            AffineRequirement("fuel_supply", "fuel", 32),
        ),
    )
    equal = AffineController(
        "Equal boiler allocation",
        tuple(
            AffineControlRule(
                n,
                "MW",
                0,
                (
                    AffineTerm("production", 1, "MW/t/h"),
                    AffineTerm("auxiliary", 0.5, "MW/MW"),
                ),
            )
            for n in ("boiler_a", "boiler_b")
        ),
        "Both boilers supply half of total demand",
    )
    return model.as_claim(domain), equal


def run_example():
    adjustable, controller = example()
    results = {
        "saved_controller": adjustable.with_controller(controller).operating_limit(
            "production"
        ),
        "adjustable_dispatch": adjustable.operating_limit("production"),
    }
    exported = {}
    for name, result in results.items():
        restored = OperatingLimitResult.from_json(result.to_json(compact=True))
        exported[name] = {
            "search": restored.to_dict(compact=True),
            "verification": verify_result(restored).to_dict(),
            "limiting_constraints": restored.limiting_constraints,
            "replay": restored.as_claim().audit_result().to_dict(compact=True)
            if restored.lower is not None
            else None,
        }
    return exported


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
