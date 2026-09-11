"""Compare equipment, uncertainty and service changes using the installed API."""

from dataclasses import replace

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    Coordinate,
    EngineeringChange,
    FiniteSet,
    LinearControl,
    LinearProcessAdapter,
    ParameterSpace,
    Scenario,
)


def example():
    space = ParameterSpace((Coordinate("load", "MW", "Synthetic steam demand"),))
    loads = FiniteSet(
        space,
        (
            Scenario("nominal", {"load": 16}, "Synthetic normal operation"),
            Scenario("peak", {"load": 20}, "Synthetic peak operation"),
        ),
    )
    model = LinearProcessAdapter(
        "Boiler with a fuel supply limit",
        space,
        outputs=(
            AffineOutput(
                "balance",
                "MW",
                "thermal_power",
                0,
                (
                    AffineTerm("boiler", 1, "MW/MW"),
                    AffineTerm("load", -1, "MW/MW"),
                ),
                "Delivered steam minus required steam",
            ),
            AffineOutput(
                "fuel",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("boiler", 1.25, "MW/MW"),),
                "Fixed 80% boiler efficiency",
            ),
        ),
        requirements=(AffineRequirement("meet_load", "balance", 0, "ge"),),
        controls=(LinearControl("boiler", "MW", 0, 18, "thermal_power"),),
        operating_limits=(AffineRequirement("fuel_supply", "fuel", 25),),
    )
    return model, loads


def run_example():
    model, loads = example()
    larger_boiler = replace(model, controls=(replace(model.controls[0], upper=20),))
    more_fuel = replace(
        model, operating_limits=(replace(model.operating_limits[0], limit=26),)
    )
    reduced_service = replace(
        model, requirements=(replace(model.requirements[0], limit=-2),)
    )
    return model.as_claim(loads).compare_changes(
        (
            EngineeringChange(
                "Larger boiler",
                larger_boiler.as_claim(loads),
                "Synthetic proposal: increase steam rating to 20 MW",
            ),
            EngineeringChange(
                "More fuel",
                more_fuel.as_claim(loads),
                "Synthetic proposal: increase fuel supply to 26 MW",
            ),
            EngineeringChange(
                "Exclude peak",
                model.as_claim(
                    FiniteSet(space=loads.space, scenarios=loads.scenarios[:1])
                ),
                "Alternative assumption: only nominal demand is admitted",
            ),
            EngineeringChange(
                "Reduce service",
                reduced_service.as_claim(loads),
                "Alternative service: permit a 2 MW steam shortfall",
            ),
        )
    )


if __name__ == "__main__":
    print(run_example().to_json(compact=True))
