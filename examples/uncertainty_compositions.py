"""Combine synthetic operating modes, physical bounds and shared restrictions.

Run: uv run python -m examples.uncertainty_compositions
With convex intersections: uv run --extra cvxpy python -m examples.uncertainty_compositions --convex
"""

import argparse
from dataclasses import replace
import json

from operange import (
    AffineOutput,
    AffineProcessAdapter,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    EllipsoidSet,
    Intersection,
    LinearConstraint,
    Parameter,
    PolytopeSet,
    Product,
    SimplexSet,
    Union,
)


def loads():
    return BoxSet(
        (
            Parameter("dryer", "MW", 10, 8, 12, 2, "Synthetic dryer envelope"),
            Parameter("evaporator", "MW", 6, 4, 8, 2, "Synthetic evaporator envelope"),
        )
    )


def header(space, capacity):
    return AffineProcessAdapter(
        "Shared steam header",
        space,
        outputs=(
            AffineOutput(
                "steam",
                "MW",
                "thermal_power",
                0,
                tuple(AffineTerm(n, 1, "MW/MW") for n in space.names),
                "Sum of process and auxiliary steam loads",
            ),
        ),
        requirements=(AffineRequirement("header_capacity", "steam", capacity),),
    )


def operating_modes_example():
    envelope = loads()
    dryer, evaporator = envelope.parameters
    modes = Union(
        (
            BoxSet(
                (
                    replace(dryer, nominal=9, upper=10),
                    replace(evaporator, nominal=7, lower=6),
                )
            ),
            BoxSet(
                (
                    replace(dryer, nominal=11, lower=10),
                    replace(evaporator, nominal=5, upper=6),
                )
            ),
        ),
        space=envelope.space,
        provenance="Two permitted production modes",
    )
    auxiliary = BoxSet(
        (Parameter("auxiliary", "MW", 1, 0.5, 1.5, 0.5, "Synthetic auxiliary demand"),)
    )
    combined = Product((modes, auxiliary))
    unrestricted = Product((envelope, auxiliary))
    return header(combined.space, 20), combined, unrestricted


def convex_examples():
    envelope = loads()
    clipped = BoxSet(
        (replace(envelope.parameters[0], upper=11), envelope.parameters[1])
    )
    correlated = EllipsoidSet(
        envelope.space,
        {
            a: {b: float(a == b) for b in envelope.space.names}
            for a in envelope.space.names
        },
    )
    joint = Intersection((clipped, correlated), backend="cvxpy")

    scheduling = PolytopeSet(
        envelope,
        (
            # Normalized deviations sum to <= 1: dryer + evaporator <= 18 MW.
            LinearConstraint("combined_production", {"dryer": 1, "evaporator": 1}, 1),
        ),
    )
    scheduled = Intersection((envelope, scheduling), backend="cvxpy")

    fractions = BoxSet(
        tuple(
            Parameter(n, "mole_fraction", 0.5, 0, 1, 1, "Synthetic binary feed")
            for n in ("a", "b")
        )
    )
    simplex = SimplexSet(fractions.space, "mole")
    specification = PolytopeSet(
        fractions,
        (
            LinearConstraint("a_fraction", {"a": 1}, 0.25),  # physical a <= .75
        ),
    )
    feed = Intersection((simplex, specification), backend="cvxpy")
    quality = AffineProcessAdapter(
        "Feed component screen",
        feed.space,
        outputs=(
            AffineOutput(
                "a_fraction",
                "mole_fraction",
                "fraction",
                0,
                (AffineTerm("a", 1, "mole_fraction/mole_fraction"),),
                "Declared mole fraction of component A",
            ),
        ),
        requirements=(AffineRequirement("downstream_limit", "a_fraction", 0.8),),
    )
    return {
        "ellipsoid_with_bounds": (header(joint.space, 18.75), joint),
        "scheduled_loads": (header(scheduled.space, 18.25), scheduled),
        "specified_feed": (quality, feed),
    }


def run_example(*, include_convex=False):
    model, combined, unrestricted = operating_modes_example()
    results = {
        "operating_modes": model.as_claim(combined).audit_result().to_dict(),
        "unrestricted_loads": model.as_claim(unrestricted).audit_result().to_dict(),
    }
    if include_convex:
        results.update(
            {
                name: model.as_claim(domain).audit_result().to_dict()
                for name, (model, domain) in convex_examples().items()
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--convex", action="store_true", help="Include CVXPY intersections"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_example(include_convex=args.convex),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
