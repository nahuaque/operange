"""Synthetic fixed-operation steam-header claims using different input domains.

Run from the repository root; stdout is the existing portable result format.
"""

import json

from operange import (
    AffineOutput,
    AffineProcessAdapter,
    AffineRequirement,
    AffineTerm,
    Axis,
    BoxSet,
    BudgetSet,
    EllipsoidSet,
    FiniteSet,
    LinearConstraint,
    PolytopeSet,
    Scenario,
    VectorParameter,
)


def example():
    loads = VectorParameter(
        "load",
        "MW",
        Axis("process", ("dryer", "evaporator")),
        nominal={"dryer": 10, "evaporator": 6},
        lower={"dryer": 8, "evaporator": 4},
        upper={"dryer": 12, "evaporator": 8},
        scale=2,
        provenance="Synthetic utility design study",
    )
    envelope = BoxSet((loads,))
    names = loads.space.names
    model = AffineProcessAdapter(
        "Shared steam header",
        loads.space,
        outputs=(
            AffineOutput(
                "steam_demand",
                "MW",
                "thermal_power",
                0,
                tuple(AffineTerm(n, 1, "MW/MW") for n in names),
                "Sum of declared process loads",
            ),
        ),
        requirements=(AffineRequirement("header_capacity", "steam_demand", 19),),
    )
    domains = {
        "box": envelope,
        "budget": BudgetSet(envelope, 1),
        "polytope": PolytopeSet(
            envelope, (LinearConstraint("joint_load", dict.fromkeys(names, 1), 1),)
        ),
        "ellipsoid": EllipsoidSet(
            loads.space, {a: {b: float(a == b) for b in names} for a in names}
        ),
        "finite": FiniteSet(
            loads.space,
            tuple(
                Scenario(
                    label, loads.flatten(values), "Synthetic listed operating case"
                )
                for label, values in (
                    ("nominal", {"dryer": 10, "evaporator": 6}),
                    ("dryer peak", {"dryer": 12, "evaporator": 6}),
                    ("evaporator peak", {"dryer": 10, "evaporator": 8}),
                )
            ),
        ),
    }
    return model, domains


def main():
    model, domains = example()
    results = {}
    for name, domain in domains.items():
        claim = model.as_claim(domain)
        results[name] = {
            "evaluation": claim.evaluate_result(domain.space.nominal).to_dict(),
            "sensitivity": claim.sensitivity_result(
                domain.space.nominal, coordinate_space="normalized"
            ).to_dict(),
            "audit": claim.audit_result().to_dict(),
        }
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
