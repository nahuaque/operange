"""One boiler/fuel equipment plan for low-load and peak production modes."""

import argparse
from dataclasses import replace
import json

from operange import AffineRequirement, BoxSet, ConvexHullSet, Parameter, Scenario
from examples.distance_and_relief import relief_example


def example():
    model, modes, changes = relief_example()
    model = replace(
        model,
        operating_limits=model.operating_limits
        + (AffineRequirement("boiler_a_minimum", "boiler_a_output", 8, "ge", 0),),
    )
    modes = replace(
        modes,
        scenarios=modes.scenarios
        + (
            Scenario(
                "low_load",
                {"dryer": 4, "evaporator": 2},
                "Declared low-production mode",
            ),
        ),
    )
    changes = changes + (
        {
            "constraint": "boiler_a_minimum",
            "maximum": 3,
            "scale": 1,
            "unit": "MW",
            "weight": 1,
        },
    )
    hull = ConvexHullSet(
        model.input_space,
        modes.scenarios,
        provenance="All convex combinations of declared production modes",
    )
    box = BoxSet(
        (
            Parameter("dryer", "MW", 10, 4, 12, 2, "Declared dryer range"),
            Parameter("evaporator", "MW", 6, 2, 8, 2, "Declared evaporator range"),
        )
    )
    return model, {"finite": modes, "hull": hull, "box": box}, changes


def run_example(*, objective="linear", backend="scipy"):
    model, domains, changes = example()
    return {
        name: model.as_claim(domain)
        .audit_result(
            backend=backend,
            relief={"changes": changes, "objective": objective, "max_scenarios": 256},
        )
        .to_dict()
        for name, domain in domains.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective", choices=("linear", "quadratic"), default="linear"
    )
    parser.add_argument("--backend", choices=("scipy", "cvxpy"), default="scipy")
    options = parser.parse_args()
    print(
        json.dumps(
            run_example(objective=options.objective, backend=options.backend),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
