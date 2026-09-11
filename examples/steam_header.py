"""A downstream utility-capacity study using only the public process API."""

from dataclasses import replace
import json

from operange import (
    AffineOutput,
    AffineProcessAdapter,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    BudgetSet,
    Parameter,
    result_from_json,
)


def example():
    loads = BoxSet(
        (
            Parameter("dryer", "MW", 10, 8, 12, 2, "Synthetic dryer envelope"),
            Parameter("evaporator", "MW", 6, 4, 8, 2, "Synthetic evaporator envelope"),
        )
    )
    # Closed-form balances for the complete declared utility boundary:
    # both users share a steam header and a boiler with fixed efficiency.
    efficiency = 0.8
    model = AffineProcessAdapter(
        name="Shared plant utilities",
        input_space=loads.space,
        outputs=(
            AffineOutput(
                "steam",
                "MW",
                "thermal_power",
                0,
                tuple(AffineTerm(n, 1, "MW/MW") for n in loads.space.names),
                "Sum of both process steam loads",
            ),
            AffineOutput(
                "fuel",
                "MW",
                "thermal_power",
                0,
                tuple(
                    AffineTerm(n, 1 / efficiency, "MW/MW") for n in loads.space.names
                ),
                "Boiler fuel balance at fixed 80% efficiency",
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
    claim = model.as_claim(loads)
    evaluation = claim.evaluate_result(loads.nominal)
    sensitivity = claim.sensitivity_result(
        loads.nominal,
        outputs=("steam", "fuel"),
        inputs=("dryer", "evaporator"),
        coordinate_space="physical",
    )
    audit = claim.audit_result()
    # Change both shared capacities, preserving the original load envelope.
    enlarged = replace(
        model,
        requirements=(
            AffineRequirement("steam_header", "steam", 21),
            AffineRequirement("boiler_fuel", "fuel", 26),
        ),
    )
    repaired = enlarged.as_claim(loads).audit_result()
    # An alternative load assumption is a separate claim, not an equipment repair.
    budgeted = model.as_claim(BudgetSet(loads, 1)).audit_result()
    unsupported = claim.breaking_result()
    restored = result_from_json(audit.to_json())
    return {
        "evaluation": evaluation.to_dict(),
        "sensitivity": sensitivity.to_dict(),
        "audit": audit.to_dict(),
        "repaired": repaired.to_dict(),
        "budgeted": budgeted.to_dict(),
        "unsupported_search": unsupported.to_dict(),
        "restored_result_id": restored.result_id,
    }


def main():
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
