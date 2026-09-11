"""Run the synthetic heat-recovery workflow without external data or solvers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path


from operange.reference import (
    FrozenHeatRecoveryContract,
    HeatDemand,
    example_claim,
    RepairAction,
    HeatRecoveryDesign,
)
from examples.reporting import (
    PricedRepair,
    compare_heat_repairs,
    render_heat_audit,
    render_threshold,
)


def example_repairs():
    return tuple(
        PricedRepair(
            RepairAction(
                name, HeatRecoveryDesign(source, power), "Synthetic equipment change"
            ),
            cost,
            "Synthetic illustrative GBP CAPEX",
        )
        for name, source, power, cost in (
            ("Baseline", 6, 2, 0),
            ("Compressor uprate", 6, 2.5, 100_000),
            ("Small source-capture upgrade", 7, 2, 150_000),
            ("Source-capture upgrade", 7.5, 2, 240_000),
            ("Larger source-capture upgrade", 8, 2, 310_000),
        )
    )


def run_example() -> tuple[str, dict, FrozenHeatRecoveryContract | None]:
    claim = example_claim()
    audit = claim.audit()
    boundary = claim.boundary_distance()
    witness = claim.breaking_witness(shortfall_margin_mw=0.01)
    repairs = compare_heat_repairs(claim, example_repairs())
    reduced = replace(claim, requirement=HeatDemand(base_heat_mw=4.8)).audit()
    policy = None
    replay = None
    if repairs.selected is not None:
        policy = claim.with_repair(repairs.selected.action.change).freeze()
        restored = FrozenHeatRecoveryContract.from_json(policy.to_json())
        replay = restored.audit_realization(claim.uncertainty.upper_corner)
    sections = [
        render_heat_audit(audit),
        "## Model and assumptions\n\n"
        "Synthetic example: all capacities, disturbance bounds, and repair costs are illustrative.\n\n"
        "- COP = 4, constant throughout the operating envelope.\n"
        "- Source capacity = 6 MW thermal; compressor capacity = 2 MW electrical.\n"
        "- Required heat = 6 * (1 + demand_increase) MW thermal.\n"
        "- Both disturbance fractions range from 0 to 0.25; their nominal values are zero.\n"
        "- Severity = max(source_derating / 0.25, demand_increase / 0.25).\n"
        "- q = 4p; source heat = q - p; source heat <= 6 * (1 - source_derating).\n"
        "- Operators observe both disturbances before selecting electrical power.\n\n",
        render_threshold(boundary),
        render_threshold(witness),
        repairs.to_markdown(),
        "## A changed service commitment\n\n"
        f"Reducing baseline delivery from 6 to 4.8 MW gives a **{reduced.status}** "
        "for the original equipment. This is a revised claim, not a repair preserving "
        "the original 6 MW commitment.\n\n",
    ]
    if replay is not None:
        sections.append(
            "## Frozen contract replay\n\n"
            f"- Selected design fingerprint: `{policy.fingerprint}`.\n"
            "- Manifest round-trip reconstructs the same explicit model and claim.\n"
            f"- Replaying the worst declared realization: **{replay.status}**.\n"
            "- This is static recourse replay. No historical calibration or causal "
            "controller backtest is performed.\n\n"
        )
    sections.append(
        "## Evidence scope\n\n"
        "SciPy/HiGHS solves the operating LP and each of two exhaustive affine failure "
        "branches. Numerical residuals, analytical capacity bounds, and independent "
        "piecewise-linear distance calculations are checked. The upper-corner audit "
        "covers this full box because delivery slack is nonincreasing in each coordinate. "
        "These arguments are specific to the constant-COP model. Solver failures remain "
        "inconclusive. Numerical tolerances apply; the report is not a probability of "
        "failure or a certificate for a real plant.\n\n"
        "Trajectories, fixed controllers, thermodynamic property models, general implicit "
        "sets, and ROmodel/PyROS integrations are outside this heat-recovery adapter. "
        "The separate storage example explores a finite two-period causal model.\n"
    )
    payload = {
        "engineering_results": {
            "audit": claim.audit_result().to_dict(),
            "boundary": claim.boundary_result().to_dict(),
            "breaking": claim.breaking_result().to_dict(),
            "sensitivity": claim.sensitivity_result(
                {"source_derating": 0.1, "demand_increase": 0.05}
            ).to_dict(),
        },
        "model_contract": asdict(claim),
        "audit": audit.to_dict(),
        "boundary": boundary.to_dict(),
        "witness": witness.to_dict(),
        "repairs": repairs.to_dict(),
        "revised_claim_audit": reduced.to_dict(),
        "selected_contract_fingerprint": policy.fingerprint if policy else None,
        "replay": replay.to_dict() if replay else None,
    }
    return "\n".join(sections), payload, policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="write Markdown here; otherwise print it"
    )
    parser.add_argument(
        "--json-output", type=Path, help="write structured run evidence"
    )
    parser.add_argument(
        "--contract-output", type=Path, help="write the selected frozen contract"
    )
    args = parser.parse_args()
    report, payload, policy = run_example()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if args.contract_output:
        if policy is None:
            parser.error("no verified passing candidate is available to freeze")
        args.contract_output.parent.mkdir(parents=True, exist_ok=True)
        args.contract_output.write_text(policy.to_json() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
