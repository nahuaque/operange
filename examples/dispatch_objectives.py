"""Minimum-fuel operation and weighted tracking of nominal boiler commands.

Run: uv run --extra cvxpy python -m examples.dispatch_objectives
"""

from dataclasses import replace
import json

from operange import ControlTarget, ControlTrackingObjective, LinearObjective
from examples.linear_dispatch import example


def run_example():
    model, loads = example()
    fuel = replace(model, objective=LinearObjective("fuel"))
    tracking = replace(
        model,
        objective=ControlTrackingObjective(
            (
                ControlTarget("boiler_a", target=8, scale=1, unit="MW"),
                ControlTarget("boiler_b", target=8, scale=1, unit="MW"),
            )
        ),
    )
    nominal = {"dryer": 10, "evaporator": 6}
    return {
        "minimum_fuel": fuel.as_claim(loads).evaluate_result(nominal).to_dict(),
        "track_nominal_commands": tracking.as_claim(loads)
        .evaluate_result(nominal, backend="cvxpy")
        .to_dict(),
        "track_at_peak": tracking.as_claim(loads)
        .evaluate_result({"dryer": 12, "evaporator": 6}, backend="cvxpy")
        .to_dict(),
        "fuel_audit": fuel.as_claim(loads).audit_result().to_dict(),
        "combined_peaks_still_infeasible": fuel.as_claim(loads)
        .evaluate_result({"dryer": 12, "evaporator": 8})
        .to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
