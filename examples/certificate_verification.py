"""Export an equipment plan and verify its certificates without optimization."""

from dataclasses import replace
import json

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    Coordinate,
    FiniteSet,
    LinearControl,
    LinearProcessAdapter,
    ParameterSpace,
    Scenario,
    result_from_json,
    verify_result,
)


def example():
    space = ParameterSpace((Coordinate("load", "MW", "Synthetic heat demand"),))
    model = LinearProcessAdapter(
        "Synthetic heater with expandable installed capacity",
        space,
        (
            AffineOutput(
                "balance",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("power", 1, "MW/MW"), AffineTerm("load", -1, "MW/MW")),
                "Delivery minus demand",
            ),
            AffineOutput(
                "delivery",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("power", 1, "MW/MW"),),
                "Heater output",
            ),
        ),
        (AffineRequirement("service", "balance", 0, "ge", 0),),
        (LinearControl("power", "MW", 0, 3),),
        (AffineRequirement("installed_capacity", "delivery", 1, tolerance=0),),
    )
    cases = FiniteSet(
        space,
        (
            Scenario("normal", {"load": 0.5}, "Synthetic mode"),
            Scenario("peak", {"load": 2}, "Synthetic mode"),
        ),
    )
    return model.as_claim(cases)


def run_example():
    claim = example()
    audit = claim.audit_result(
        relief={
            "changes": [
                {
                    "constraint": "installed_capacity",
                    "maximum": 2,
                    "scale": 1,
                    "unit": "MW",
                }
            ],
            "objective": "linear",
        }
    )
    exported = audit.to_json(compact=True)
    report = verify_result(
        exported, expected_contract_id=claim.contract.ref.artifact_id
    )
    failure = claim.evaluate_result({"load": 2})
    proofs = []
    for proof in failure.evidence:
        if proof.evidence_id == "recourse":
            data = proof.to_dict()["details"]
            data["contradiction_gap_exact"] = "0"
            proof = replace(proof, details=data)
        proofs.append(proof)
    # Re-exporting gives the tampered evidence a consistent content identity.
    tampered = replace(failure, evidence=tuple(proofs))
    loaded = result_from_json(tampered.to_json())
    rejected = verify_result(loaded)
    return {
        "audit": audit.to_dict(compact=True),
        "verification": report.to_dict(),
        "tampered_verification": rejected.to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
