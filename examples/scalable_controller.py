"""Synthesize a steam controller over 20 independent loads without corners."""

import json

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    FrozenController,
    LinearControl,
    LinearObjective,
    LinearProcessAdapter,
    Parameter,
    verify_result,
)


def example():
    domain = BoxSet(
        tuple(
            Parameter(f"load_{i:02d}", "MW", 0.875, 0.75, 1, 0.125, "Synthetic load")
            for i in range(20)
        )
    )
    model = LinearProcessAdapter(
        name="Twenty steam users sharing two boilers",
        input_space=domain.space,
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
                (AffineTerm("boiler_a", 1, "MW/MW"), AffineTerm("boiler_b", 1, "MW/MW"))
                + tuple(AffineTerm(n, -1, "MW/MW") for n in domain.space.names),
                "Boiler delivery minus the twenty process loads",
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
                "Boiler A efficiency 80%; boiler B efficiency 50%",
            ),
        ),
        requirements=(AffineRequirement("meet_load", "steam_balance", 0, "ge"),),
        operating_limits=(
            AffineRequirement("no_excess_steam", "steam_balance", 0),
            AffineRequirement("shared_fuel", "fuel", 32),
        ),
        objective=LinearObjective("fuel"),
    )
    return model.as_claim(domain)


def run_example():
    claim = example()
    # The box has 1,048,576 corners. Linear box synthesis does not enumerate any.
    search = claim.synthesize_controller(name="Twenty-load steam controller")
    result = {
        "input_count": len(claim.domain.space.names),
        "box_corner_count": 2 ** len(claim.domain.space.names),
        "synthesis": search.to_dict(compact=True),
    }
    if search.verified:
        frozen = FrozenController.from_json(search.freeze().to_json())
        result.update(
            frozen_controller=frozen.to_dict(),
            verification=verify_result(search.audit.to_json(compact=True)).to_dict(),
            interior=frozen.evaluate_result(
                {n: 0.875 for n in claim.domain.space.names}
            ).to_dict(compact=True),
        )
    return result


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
