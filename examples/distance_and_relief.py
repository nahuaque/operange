"""Scaled convex distances and explicit combinations of operating-limit relief.

Run: uv run --extra cvxpy python -m examples.distance_and_relief
"""

from dataclasses import replace
import json

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    NormalizedL2,
    NormalizedLInf,
)
from examples.linear_dispatch import example as dispatch_example
from examples.uncertainty_compositions import convex_examples


def distance_example():
    model, domain = convex_examples()["ellipsoid_with_bounds"]
    model = replace(model, requirements=(replace(model.requirements[0], limit=18.7),))
    return model, domain


def relief_example():
    model, loads = dispatch_example()
    # The control box bounds the contemplated study. The original installed
    # capacity remains an explicit operating limit that relief may change.
    model = replace(
        model,
        controls=(replace(model.controls[0], upper=14), model.controls[1]),
        outputs=model.outputs
        + (
            AffineOutput(
                "boiler_a_output",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("boiler_a", 1, "MW/MW"),),
                "Boiler A steam output",
            ),
        ),
        requirements=tuple(replace(r, tolerance=0) for r in model.requirements),
        operating_limits=tuple(replace(r, tolerance=0) for r in model.operating_limits)
        + (AffineRequirement("boiler_a_capacity", "boiler_a_output", 12, tolerance=0),),
    )
    changes = (
        {
            "constraint": "boiler_a_capacity",
            "maximum": 2,
            "scale": 1,
            "unit": "MW",
            "weight": 1,
        },
        {
            "constraint": "shared_fuel",
            "maximum": 2,
            "scale": 1,
            "unit": "MW",
            "weight": 1,
        },
    )
    return model, loads, changes


def run_example():
    model, domain = distance_example()
    results = {}
    for name, metric in (("linf", NormalizedLInf), ("euclidean", NormalizedL2)):
        claim = model.as_claim(domain, distance=metric(domain.space))
        results[name] = claim.breaking_result(
            backend="cvxpy",
            violation_margins={"header_capacity": 0.01},
        ).to_dict()
    model, loads, changes = relief_example()
    for objective in ("linear", "quadratic"):
        result = model.as_claim(loads).evaluate_result(
            {"dryer": 12, "evaporator": 8},
            backend="cvxpy",
            relief={"changes": changes, "objective": objective},
        )
        results[f"{objective}_relief"] = result.to_dict()
        proposal = next(
            e.details["candidate"] for e in result.evidence if e.evidence_id == "relief"
        )
        limits = {c["constraint"]: c["changed_limit"] for c in proposal["changes"]}
        changed = replace(
            model,
            operating_limits=tuple(
                replace(r, limit=limits.get(r.name, r.limit))
                for r in model.operating_limits
            ),
        )
        results[f"{objective}_reaudit"] = (
            changed.as_claim(loads).audit_result(backend="cvxpy").to_dict()
        )
    return results


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
