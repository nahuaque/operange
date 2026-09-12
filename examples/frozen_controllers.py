"""Freeze and replay two boiler allocation rules using only the installed API."""

import json
from dataclasses import replace

from operange import (
    AffineControlRule,
    AffineController,
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    Coordinate,
    EngineeringChange,
    FiniteSet,
    FrozenController,
    LinearControl,
    LinearProcessAdapter,
    NormalizedLInf,
    Parameter,
    ParameterSpace,
    Scenario,
)


def example():
    space = ParameterSpace(
        tuple(
            Coordinate(n, "MW", "Synthetic process steam demand")
            for n in ("dryer", "evaporator")
        )
    )
    domain = FiniteSet(
        space,
        tuple(
            Scenario(
                name, {"dryer": a, "evaporator": b}, "Synthetic finite operating case"
            )
            for name, a, b in (
                ("nominal", 10, 6),
                ("dryer peak", 12, 6),
                ("evaporator peak", 10, 8),
                ("combined", 12, 8),
            )
        ),
    )
    model = LinearProcessAdapter(
        "Two boilers sharing a 32 MW fuel supply",
        space,
        outputs=(
            AffineOutput(
                "steam_balance",
                "MW",
                "thermal_power",
                0,
                tuple(
                    AffineTerm(n, coefficient, "MW/MW")
                    for n, coefficient in (
                        ("boiler_a", 1),
                        ("boiler_b", 1),
                        ("dryer", -1),
                        ("evaporator", -1),
                    )
                ),
                "Delivered steam minus both process demands",
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
                "Fixed efficiencies: boiler A 80%, boiler B 50%",
            ),
        ),
        requirements=(AffineRequirement("meet_load", "steam_balance", 0, "ge"),),
        controls=(
            LinearControl("boiler_a", "MW", 0, 12, "thermal_power"),
            LinearControl("boiler_b", "MW", 0, 10, "thermal_power"),
        ),
        operating_limits=(
            AffineRequirement("no_excess_steam", "steam_balance", 0),
            AffineRequirement("shared_fuel", "fuel", 32),
        ),
    )
    return model, domain


def allocation(name, a, b):
    return AffineController(
        name,
        tuple(
            AffineControlRule(
                control,
                "MW",
                0,
                tuple(
                    AffineTerm(observation, share, "MW/MW")
                    for observation in ("dryer", "evaporator")
                ),
            )
            for control, share in (("boiler_a", a), ("boiler_b", b))
        ),
        "Synthetic caller-supplied allocation; coefficients are fixed without fitting",
    )


def run_example():
    model, domain = example()
    equal = model.as_claim(domain, controller=allocation("50/50 allocation", 0.5, 0.5))
    efficient = model.as_claim(
        domain, controller=allocation("60/40 allocation", 0.6, 0.4)
    )
    comparison = equal.compare_changes(
        (
            EngineeringChange(
                "60/40 allocation", efficient, "Change the operating rule"
            ),
            EngineeringChange(
                "Adjustable benchmark",
                model.as_claim(domain),
                "Separate audit of independently adjustable commands",
            ),
        )
    )
    frozen = efficient.freeze()
    loaded = FrozenController.from_json(frozen.to_json())
    held_out = FiniteSet(
        domain.space,
        (
            Scenario(
                "new 17 MW load", {"dryer": 11, "evaporator": 6}, "Held-out finite case"
            ),
            Scenario(
                "new 19 MW load", {"dryer": 11, "evaporator": 8}, "Held-out finite case"
            ),
        ),
    )
    return {
        "comparison": comparison.to_dict(compact=True),
        "frozen_controller": frozen.to_dict(),
        "replayed_audit": loaded.audit_result().to_dict(compact=True),
        "held_out_audit": loaded.as_claim(held_out)
        .audit_result()
        .to_dict(compact=True),
        "continuous_envelope": run_envelope_example(),
    }


def envelope_example():
    model, _ = example()
    domain = BoxSet(
        (
            Parameter("dryer", "MW", 10, 8, 12, 2, "Declared continuous load envelope"),
            Parameter(
                "evaporator", "MW", 6, 4, 8, 2, "Declared continuous load envelope"
            ),
        )
    )
    model = replace(
        model,
        input_space=domain.space,
        operating_limits=tuple(
            replace(r, limit=30) if r.name == "shared_fuel" else r
            for r in model.operating_limits
        ),
    )
    return model.as_claim(
        domain,
        controller=allocation("60/40 allocation", 0.6, 0.4),
        distance=NormalizedLInf(domain.space),
    )


def run_envelope_example():
    claim = envelope_example()
    audit = claim.audit_result()
    point = audit.payload.witness.realizations[0]
    # Diagnose physically adjustable dispatch at the controller's failing point.
    # This is a separate question, with explicitly fully observed permissions.
    diagnostic = claim.adapter.model.as_claim(claim.domain).evaluate_result(
        point,
        diagnose=True,
        relief={"constraint": "shared_fuel", "maximum": 2},
    )
    upgraded = replace(
        claim.adapter.model,
        operating_limits=tuple(
            replace(r, limit=32) if r.name == "shared_fuel" else r
            for r in claim.adapter.model.operating_limits
        ),
    )
    changed = upgraded.as_claim(
        claim.domain,
        controller=claim.adapter.controller,
        recourse=claim.recourse,
        distance=claim.distance,
    )
    comparison = claim.compare_changes(
        (
            EngineeringChange(
                "32 MW fuel supply",
                changed,
                "Caller-supplied equipment change preserving the load envelope and controller",
            ),
        )
    )
    frozen = changed.freeze()
    loaded = FrozenController.from_json(frozen.to_json())
    return {
        "audit": audit.to_dict(compact=True),
        "breaking": claim.breaking_result(
            constraints=("shared_fuel",), violation_margins={"shared_fuel": 0.01}
        ).to_dict(compact=True),
        "diagnosis": diagnostic.to_dict(compact=True),
        "comparison": comparison.to_dict(compact=True),
        "frozen_controller": frozen.to_dict(),
        "replayed_audit": loaded.audit_result().to_dict(compact=True),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
