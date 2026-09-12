"""Adjustable boiler operation over a continuous load envelope."""

import argparse
from dataclasses import replace
import json

from operange import BoxSet, ConvexHullSet, Parameter
from examples.linear_dispatch import example


def run_example(*, backend="scipy"):
    model, cases = example()
    envelope = BoxSet(
        (
            Parameter("dryer", "MW", 10, 10, 12, 2, "Declared dryer operating range"),
            Parameter(
                "evaporator", "MW", 6, 6, 8, 2, "Declared evaporator operating range"
            ),
        )
    )
    scheduled = ConvexHullSet(
        model.input_space,
        tuple(s for s in cases.scenarios if s.name != "combined"),
        provenance="Scheduling permits convex combinations of nominal and individual peak modes",
    )
    enlarged = replace(
        model,
        operating_limits=tuple(
            replace(limit, limit=32) if limit.name == "shared_fuel" else limit
            for limit in model.operating_limits
        ),
    )
    unrestricted = model.as_claim(envelope)
    failed = unrestricted.audit_result(backend=backend, max_vertices=4)
    return {
        "unrestricted_box": failed.to_dict(),
        "scheduled_hull": model.as_claim(scheduled)
        .audit_result(backend=backend)
        .to_dict(),
        "enlarged_fuel_supply": enlarged.as_claim(envelope)
        .audit_result(backend=backend)
        .to_dict(),
        "interior_dispatch": model.as_claim(scheduled)
        .evaluate_result(
            {"dryer": 11, "evaporator": 7},
            backend=backend,
        )
        .to_dict(),
        "failure_replay": unrestricted.evaluate_result(
            failed.payload.witness.realizations[0],
            backend=backend,
        ).to_dict(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("scipy", "cvxpy"), default="scipy")
    print(
        json.dumps(
            run_example(backend=parser.parse_args().backend),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
