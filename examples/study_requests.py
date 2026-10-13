"""Inspect exact study support before executing a certified linear audit."""

import json

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    AuditRequest,
    BoxSet,
    ControllerSynthesisRequest,
    DispatchChange,
    DispatchSensitivityRequest,
    LinearControl,
    LinearObjective,
    LinearProcessAdapter,
    OperatingSpec,
    Parameter,
    verify_result,
)


def example():
    loads = BoxSet((Parameter("load", "MW", 1, 0, 2, 1, "Declared load range"),))
    model = LinearProcessAdapter(
        "Adjustable utility supply",
        loads.space,
        (
            AffineOutput(
                "delivery",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("fuel", 1, "MW/MW"), AffineTerm("load", -1, "MW/MW")),
                "Delivered power minus load",
            ),
            AffineOutput(
                "fuel_use",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("fuel", 1, "MW/MW"),),
                "Fuel consumption",
            ),
        ),
        (AffineRequirement("service", "delivery", 0, "ge"),),
        (LinearControl("fuel", "MW", 0, 4),),
        (AffineRequirement("capacity", "fuel_use", 3),),
        objective=LinearObjective("fuel_use"),
    )
    return model.as_claim(loads)


def run_example():
    claim = example()
    request = AuditRequest(
        required_evidence=("continuous_coverage", "independent_certificate")
    )
    capability = claim.check_study(request)
    audit = claim.run_study(request)
    sensitivity = claim.run_study(
        DispatchSensitivityRequest(
            {"load": 1},
            (DispatchChange("input", "load", "MW", 0.25),),
        )
    )
    shared = claim.adapter.as_claim(
        claim.domain,
        operating=OperatingSpec.shared(claim.adapter.controls),
    )
    return {
        "audit_capability": capability.to_dict(),
        "audit": audit.to_dict(),
        "verification": verify_result(audit).to_dict(),
        "sensitivity": sensitivity.to_dict(),
        "budget_capability": claim.check_study(AuditRequest(max_vertices=1)).to_dict(),
        "controller_capability": shared.check_study(
            ControllerSynthesisRequest()
        ).to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2))
