"""Compare fixed, causal, and perfect-foresight thermal-storage decisions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path


from operange.reference import (
    example_storage_claim,
    StorageRepairAction,
    StorageDesign,
)
from examples.reporting import (
    PricedRepair,
    compare_storage_repairs,
    render_storage_audit,
)


def example_storage_repairs():
    provenance = "Synthetic equipment/information change"
    costs = "Synthetic illustrative one-time GBP CAPEX"
    return (
        PricedRepair(StorageRepairAction("Baseline", provenance), 0, costs),
        PricedRepair(
            StorageRepairAction(
                "More charging power",
                provenance,
                design=StorageDesign(charging_power_mw=2),
            ),
            10_000,
            costs,
        ),
        PricedRepair(
            StorageRepairAction(
                "Earlier event notification",
                provenance + "; exact advance signal",
                reveal_before_preparation=True,
            ),
            15_000,
            costs,
        ),
        PricedRepair(
            StorageRepairAction(
                "Larger store", provenance, design=StorageDesign(capacity_mwh=2)
            ),
            40_000,
            costs,
        ),
    )


def run_storage_example() -> tuple[str, dict]:
    claim = example_storage_claim()
    fixed = claim.audit("fixed", fixed_preparation_power_mw=0.5)
    causal = claim.audit("causal")
    foresight = claim.audit("perfect_foresight")
    repairs = compare_storage_repairs(claim, example_storage_repairs())
    sections = [
        "# Thermal storage: the value of timely information\n\n"
        "Synthetic two-period example. All equipment values and repair costs are illustrative.\n\n"
        "| Operating contract | Verdict |\n| --- | --- |\n"
        f"| Fixed preparation rule (0.5 MW) | {fixed.status} |\n"
        f"| Best causal policy under declared observations | {causal.status} |\n"
        f"| Perfect-foresight benchmark | {foresight.status} |\n\n",
        "## Physical and information contract\n\n"
        "- Preparation lasts one hour. The store starts empty, holds at most 1 MWh, "
        "and can charge from an available external heat source at up to 1 MW.\n"
        "- The next one-hour event is either 1 MW of heat demand or 1 MW of mandatory "
        "recovered heat. The event is revealed after preparation, before event actions.\n"
        "- Heat must exactly balance. Unmet demand and discarded recovered heat are prohibited.\n"
        "- Event charging and discharging are each limited to 1 MW. Simultaneous "
        "charging/discharging is prohibited. Efficiencies are one; standing loss is zero.\n"
        "- Final energy may lie between zero and installed capacity. This is a finite-horizon "
        "service claim, with no cyclic reset or indefinite-operation guarantee.\n\n"
        "The demand future requires 1 MWh prepared. The surplus future requires zero "
        "prepared energy to leave room for incoming heat. Each future can be handled "
        "with advance knowledge; their shared preparation history creates the conflict.\n\n",
        render_storage_audit(fixed),
        render_storage_audit(causal),
        render_storage_audit(foresight),
        repairs.to_markdown(),
    ]
    if repairs.selected:
        action = repairs.selected.action.change
        explanation = (
            "The early notification is an explicit information intervention. Actions may "
            "differ because the event identities are now observable before preparation."
            if action.reveal_before_preparation
            else "This equipment repair retains the original observation timing. "
            "Futures with the same preparation observation share their preparation action."
        )
        sections.extend(
            [
                f"## Selected repair's causal plan\n\n{explanation}\n\n",
                render_storage_audit(repairs.selected.causal),
            ]
        )
    verified_comparison = (
        (fixed.status, causal.status, foresight.status) == ("fail", "fail", "pass")
        and [r.causal.status for r in repairs.candidates]
        == ["fail", "fail", "pass", "pass"]
        and all(r.perfect_foresight.status == "pass" for r in repairs.candidates)
    )
    sections.append("## What this establishes\n\n")
    sections.append(
        (
            "The perfect-foresight benchmark would accept the baseline equipment. Under the "
            "original information timing, a causal plan needs a repair. In this synthetic "
            "catalog, earlier exact notification is cheaper than enlarging the store. "
            "More charging power alone cannot create the missing storage headroom.\n\n"
            if verified_comparison
            else "The example comparison was not fully verified in this run. Use the "
            "individual verdicts and solver evidence above; unresolved outcomes do not "
            "establish feasibility, impossibility, or the cheapest feasible repair.\n\n"
        )
        + "The LP uses a single preparation decision for each observable history. An "
        "independent interval calculation checks the full set of feasible preparation "
        "states. Its conflicting lower and upper bounds identify a minimal pair of "
        "futures. Both LP evidence and the analytical check must agree before the "
        "report declares pass or fail; solver failures remain inconclusive.\n\n"
        "Coverage is the declared finite tree and discrete two-period model. There is "
        "no failure probability, continuous-domain robustness radius, or claim about "
        "an unmodeled future. Exported operating plans are review evidence, not a "
        "deployed controller or a historical backtest.\n"
    )
    return "\n".join(sections), {
        "engineering_results": {
            "fixed": claim.audit_result(
                "fixed", fixed_preparation_power_mw=0.5
            ).to_dict(),
            "causal": claim.audit_result("causal").to_dict(),
            "perfect_foresight": claim.audit_result("perfect_foresight").to_dict(),
        },
        "claim": asdict(claim),
        "fixed": fixed.to_dict(),
        "causal": causal.to_dict(),
        "perfect_foresight": foresight.to_dict(),
        "repairs": repairs.to_dict(),
        "selected_repair": repairs.selected.action.change.name
        if repairs.selected
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="write the Markdown comparison report"
    )
    parser.add_argument(
        "--json-output", type=Path, help="write the full model, decisions, and evidence"
    )
    args = parser.parse_args()
    report, evidence = run_storage_example()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
