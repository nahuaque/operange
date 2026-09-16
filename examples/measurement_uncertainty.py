"""A steam controller acts on a biased meter, not on the true load."""

from dataclasses import replace
import json

from operange import (
    AffineControlRule,
    AffineController,
    AffineMeasurement,
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    FrozenController,
    LinearControl,
    LinearObjective,
    LinearProcessAdapter,
    MeasurementModel,
    Parameter,
    verify_result,
)


def example():
    domain = BoxSet(
        (
            Parameter("load", "MW", 9, 8, 10, 1, "True steam demand"),
            Parameter("meter_bias", "MW", 0, -0.25, 0.25, 0.25, "Unknown meter offset"),
        )
    )
    measurements = MeasurementModel(
        "Steam demand meter",
        (
            AffineMeasurement(
                "measured_load",
                "MW",
                0,
                (
                    AffineTerm("load", 1, "MW/MW"),
                    AffineTerm("meter_bias", 1, "MW/MW"),
                ),
                "Measured demand equals true demand plus unknown bias",
            ),
        ),
        error_inputs=("meter_bias",),
    )
    model = LinearProcessAdapter(
        "Steam service with an imperfect demand meter",
        domain.space,
        (
            AffineOutput(
                "steam_balance",
                "MW",
                "thermal_power",
                0,
                (
                    AffineTerm("steam", 1, "MW/MW"),
                    AffineTerm("load", -1, "MW/MW"),
                ),
                "Delivery minus true demand",
            ),
            AffineOutput(
                "fuel",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("steam", 1.25, "MW/MW"),),
                "Boiler efficiency 80%",
            ),
        ),
        (AffineRequirement("serve_true_load", "steam_balance", 0, "ge"),),
        (LinearControl("steam", "MW", 0, 11),),
        (AffineRequirement("excess_steam", "steam_balance", 0.5),),
        objective=LinearObjective("fuel"),
        measurements=measurements,
    )
    return model.as_claim(domain)


def run_example():
    claim = example()
    naive = AffineController(
        "Follow the meter",
        (
            AffineControlRule(
                "steam", "MW", 0, (AffineTerm("measured_load", 1, "MW/MW"),)
            ),
        ),
        "No allowance for underreading",
    )
    baseline = claim.with_controller(naive).audit_result()
    search = claim.synthesize_controller(name="Allow for meter error")
    result = {
        "naive_audit": baseline.to_dict(compact=True),
        "synthesis": search.to_dict(compact=True),
    }
    if search.verified:
        frozen = FrozenController.from_json(search.freeze().to_json())
        wider = BoxSet(
            (
                claim.domain.parameter("load"),
                replace(claim.domain.parameter("meter_bias"), lower=-0.5, upper=0.5),
            )
        )
        result.update(
            frozen_controller=frozen.to_dict(),
            replay=frozen.evaluate_result({"load": 9, "meter_bias": -0.25}).to_dict(
                compact=True
            ),
            verification=verify_result(search.audit).to_dict(),
            larger_error_audit=frozen.as_claim(wider)
            .audit_result()
            .to_dict(compact=True),
        )
    return result


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
