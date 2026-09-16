"""Marginal fuel use, a boiler-capacity upgrade and a blocked upgrade."""

from dataclasses import replace
import json

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    DispatchChange,
    LinearControl,
    LinearObjective,
    LinearProcessAdapter,
    Parameter,
    verify_result,
)


def example():
    domain = BoxSet((Parameter("load", "MW", 16, 0, 22, 4, "Declared steam demand"),))
    model = LinearProcessAdapter(
        "Two boilers with explicit marginal fuel costs",
        domain.space,
        outputs=(
            AffineOutput(
                "balance",
                "MW",
                "thermal_power",
                0,
                (
                    AffineTerm("a", 1, "MW/MW"),
                    AffineTerm("b", 1, "MW/MW"),
                    AffineTerm("load", -1, "MW/MW"),
                ),
                "Steam delivery equals load",
            ),
            AffineOutput(
                "fuel",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("a", 1.25, "MW/MW"), AffineTerm("b", 2, "MW/MW")),
                "Fixed marginal fuel coefficients",
            ),
            AffineOutput(
                "a_output",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("a", 1, "MW/MW"),),
                "Boiler A steam delivery",
            ),
        ),
        requirements=(AffineRequirement("serve", "balance", 0, "ge", tolerance=0),),
        controls=(LinearControl("a", "MW", 0, 12), LinearControl("b", "MW", 0, 10)),
        operating_limits=(
            AffineRequirement("no_excess", "balance", 0, tolerance=0),
            AffineRequirement("fuel_supply", "fuel", 32, tolerance=0),
        ),
        objective=LinearObjective("fuel"),
    )
    return model, domain


def run_example():
    model, domain = example()
    claim = model.as_claim(domain)
    load = DispatchChange("input", "load", "MW", step=2)
    capacity = DispatchChange("control_upper", "a", "MW", step=2)
    blocked = replace(
        model,
        operating_limits=(
            *model.operating_limits,
            AffineRequirement("burner_limit", "a_output", 12, tolerance=0),
        ),
    )
    results = {
        "nominal": claim.dispatch_sensitivity(
            {"load": 16},
            changes=(
                load,
                capacity,
                DispatchChange("operating_limit", "fuel_supply", "MW"),
            ),
        ),
        "boiler_threshold": claim.dispatch_sensitivity({"load": 12}, changes=(load,)),
        "blocked_upgrade": blocked.as_claim(domain).dispatch_sensitivity(
            {"load": 16}, changes=(capacity,)
        ),
        "fuel_boundary": claim.dispatch_sensitivity({"load": 20.5}, changes=(load,)),
    }
    return {
        name: {
            "sensitivity": r.to_dict(compact=True),
            "verification": verify_result(r).to_dict(),
        }
        for name, r in results.items()
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
