"""Optional prepared dispatch and a joint convex uncertainty envelope.

Run: uv run --extra cvxpy python -m examples.convex_backends
"""

import json

from operange import (
    AffineOutput,
    AffineProcessAdapter,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    EllipsoidSet,
    Intersection,
    Parameter,
)
from examples.linear_dispatch import example as dispatch_example


def intersection_example():
    envelope = BoxSet(
        (
            Parameter("dryer", "MW", 10, 8, 11, 2, "Synthetic dryer envelope"),
            Parameter("evaporator", "MW", 6, 4, 8, 2, "Synthetic evaporator envelope"),
        )
    )
    correlated = EllipsoidSet(
        envelope.space,
        {
            a: {b: float(a == b) for b in envelope.space.names}
            for a in envelope.space.names
        },
    )
    joint = Intersection((envelope, correlated), backend="cvxpy")
    model = AffineProcessAdapter(
        "Shared steam header",
        joint.space,
        outputs=(
            AffineOutput(
                "steam_demand",
                "MW",
                "thermal_power",
                0,
                tuple(AffineTerm(n, 1, "MW/MW") for n in joint.space.names),
                "Sum of process steam loads",
            ),
        ),
        requirements=(AffineRequirement("header_capacity", "steam_demand", 18.75),),
    )
    return model, joint


def run_example():
    boilers, loads = dispatch_example()
    claim = boilers.as_claim(loads)
    model, joint = intersection_example()
    return {
        "dispatch": claim.audit_result(backend="cvxpy").to_dict(),
        "relief": claim.evaluate_result(
            {"dryer": 12, "evaporator": 8},
            backend="cvxpy",
            diagnose=True,
            relief={"constraint": "shared_fuel", "maximum": 2},
        ).to_dict(),
        "box": model.as_claim(joint.factors[0]).audit_result().to_dict(),
        "ellipsoid": model.as_claim(joint.factors[1]).audit_result().to_dict(),
        "intersection": model.as_claim(joint).audit_result().to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
