"""Two dispatchable boilers, uncertain loads and a shared fuel constraint."""

from dataclasses import replace
import json

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    Coordinate,
    DecisionRule,
    FiniteSet,
    LinearControl,
    LinearProcessAdapter,
    ParameterSpace,
    RecoursePolicy,
    Scenario,
)


def example():
    space = ParameterSpace(
        tuple(
            Coordinate(name, "MW", "Synthetic process steam demand")
            for name in ("dryer", "evaporator")
        )
    )
    loads = FiniteSet(
        space,
        (
            Scenario(
                "nominal", {"dryer": 10, "evaporator": 6}, "Synthetic nominal case"
            ),
            Scenario(
                "dryer_peak", {"dryer": 12, "evaporator": 6}, "Synthetic dryer peak"
            ),
            Scenario(
                "evaporator_peak",
                {"dryer": 10, "evaporator": 8},
                "Synthetic evaporator peak",
            ),
            Scenario(
                "combined",
                {"dryer": 12, "evaporator": 8},
                "Synthetic simultaneous peaks",
            ),
        ),
    )
    model = LinearProcessAdapter(
        name="Two boilers sharing a fuel supply",
        input_space=space,
        controls=(
            LinearControl("boiler_a", "MW", 0, 12, "thermal_power"),
            LinearControl("boiler_b", "MW", 0, 10, "thermal_power"),
        ),
        outputs=(
            AffineOutput(
                "steam_balance",
                "MW",
                "thermal_power",
                0,
                tuple(
                    AffineTerm(n, coefficient, "MW/MW")
                    for n, coefficient in (
                        ("boiler_a", 1),
                        ("boiler_b", 1),
                        ("dryer", -1),
                        ("evaporator", -1),
                    )
                ),
                "Boiler delivery minus both process loads",
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
                "Fixed efficiencies: boiler A 80%, boiler B 50%",
            ),
        ),
        requirements=(AffineRequirement("meet_load", "steam_balance", 0, "ge"),),
        operating_limits=(
            AffineRequirement("no_excess_steam", "steam_balance", 0),
            AffineRequirement("shared_fuel", "fuel", 30),
        ),
    )
    return model, loads


def run_example():
    model, loads = example()
    fixed = RecoursePolicy(
        "fixed",
        (
            DecisionRule("boiler_a", "MW", "operation", fixed_value=10),
            DecisionRule("boiler_b", "MW", "operation", fixed_value=6),
        ),
    )
    enlarged = replace(
        model,
        operating_limits=tuple(
            replace(limit, limit=32) if limit.name == "shared_fuel" else limit
            for limit in model.operating_limits
        ),
    )
    return {
        "nominal": model.as_claim(loads)
        .evaluate_result({"dryer": 10, "evaporator": 6})
        .to_dict(),
        "fixed": model.as_claim(loads, recourse=fixed).audit_result().to_dict(),
        "adjustable": model.as_claim(loads).audit_result().to_dict(),
        "enlarged_fuel_supply": enlarged.as_claim(loads).audit_result().to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
