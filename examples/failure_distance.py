"""Boundary and positive-violation distances through the public affine API."""

import json

from operange import (
    AffineOutput,
    AffineProcessAdapter,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    LinearConstraint,
    NormalizedLInf,
    Parameter,
    PolytopeSet,
)


def example():
    loads = BoxSet(
        (
            Parameter("dryer", "MW", 10, 8, 12, 2, "Synthetic dryer envelope"),
            Parameter("evaporator", "MW", 6, 4, 8, 2, "Synthetic evaporator envelope"),
        )
    )
    model = AffineProcessAdapter(
        name="Shared utility failure distance",
        input_space=loads.space,
        outputs=(
            AffineOutput(
                "steam",
                "MW",
                "thermal_power",
                0,
                tuple(AffineTerm(n, 1, "MW/MW") for n in loads.space.names),
                "Sum of both process loads",
            ),
            AffineOutput(
                "fuel",
                "MW",
                "thermal_power",
                0,
                tuple(AffineTerm(n, 1.25, "MW/MW") for n in loads.space.names),
                "Fuel at fixed 80% efficiency",
            ),
        ),
        requirements=(
            AffineRequirement("steam_header", "steam", 19),
            AffineRequirement("boiler_fuel", "fuel", 24),
        ),
    )
    return model, loads


def run_example():
    model, loads = example()
    distance = NormalizedLInf(loads.space)
    claim = model.as_claim(loads, distance=distance)
    # This additional load restriction changes the declared uncertainty domain.
    restricted = PolytopeSet(loads, (LinearConstraint("dryer_cap", {"dryer": 1}, 0.5),))
    constrained = model.as_claim(restricted, distance=distance)
    return {
        "boundary": claim.boundary_result().to_dict(),
        "breaking": claim.breaking_result(
            violation_margins={"steam_header": 0.01, "boiler_fuel": 0.01},
        ).to_dict(),
        "restricted_boundary": constrained.boundary_result().to_dict(),
        "restricted_breaking": constrained.breaking_result(
            violation_margins={"steam_header": 0.01, "boiler_fuel": 0.01},
        ).to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
