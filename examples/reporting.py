"""Consumer-owned pricing, selection, and rendering for reference examples."""

from dataclasses import asdict, dataclass
import json
from math import isfinite
from typing import Any
from operange.reference import (
    ClaimAudit,
    ThresholdResult,
    StorageAudit,
    RepairAction,
    StorageRepairAction,
)


@dataclass(frozen=True)
class PricedRepair:
    change: RepairAction | StorageRepairAction
    added_capex_gbp: float
    cost_provenance: str

    def __post_init__(self):
        if not isinstance(self.change, (RepairAction, StorageRepairAction)):
            raise ValueError("change must be a reference-model engineering repair")
        if (
            isinstance(self.added_capex_gbp, bool)
            or not isinstance(self.added_capex_gbp, (int, float))
            or not isfinite(self.added_capex_gbp)
            or self.added_capex_gbp < 0
        ):
            raise ValueError("added_capex_gbp must be finite and nonnegative")
        if (
            not isinstance(self.cost_provenance, str)
            or not self.cost_provenance.strip()
        ):
            raise ValueError("cost_provenance must be nonempty")


LABELS = {
    "source_capacity": "Available source heat",
    "compressor_capacity": "Compressor power",
    "source_derating": "Source-capacity derating",
    "demand_increase": "Demand increase",
}


def number(value: float | None) -> str:
    return "unresolved" if value is None else f"{0.0 if value == 0 else value:.6f}"


def cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


class Artifact:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)


def render_heat_audit(audit: ClaimAudit) -> str:
    return "\n".join(
        [
            f"# {audit.claim_name}",
            "",
            f"Claim verdict: **{audit.status}** (within the stated numerical tolerance).",
            "",
            f"- Coverage: {audit.coverage}.",
            f"- Recourse: {audit.recourse}.",
            f"- Nominal maximum delivery: {number(audit.nominal.maximum_heat_mw)} MW thermal.",
            f"- Worst maximum delivery: {number(audit.worst.maximum_heat_mw)} MW thermal.",
            f"- Worst required delivery: {number(audit.worst.required_heat_mw)} MW thermal.",
            f"- Minimum delivery slack: {number(audit.worst.slack_mw)} MW thermal.",
            f"- Binding limits: {', '.join(LABELS[name] for name in audit.worst.binding_limits) or 'unresolved'}.",
            f"- Numerical tolerance: {audit.worst.evidence.tolerance:g} (MW residuals).",
            f"- Solve evidence: {audit.worst.evidence.status}; {audit.worst.evidence.message}",
            "",
        ]
    )


def render_threshold(threshold: ThresholdResult) -> str:
    severity = (
        "- Severity: not attainable in the declared domain."
        if threshold.status == "unreachable"
        else f"- {'Minimum' if threshold.minimum_verified else 'Candidate'} severity: {number(threshold.distance)}."
    )
    lines = [
        f"## {'Breaking witness' if threshold.kind == 'witness' else 'Feasibility boundary'}",
        "",
        f"- Status: {threshold.status}.",
        severity,
        f"- Distance: {threshold.distance_metric}.",
        f"- Required shortfall: {threshold.shortfall_margin_mw:g} MW thermal.",
        "",
    ]
    if threshold.status == "already_broken":
        lines.extend(
            ["The nominal realization already fails; no disturbance is needed.", ""]
        )
    elif threshold.kind == "boundary":
        lines.extend(
            [
                "A zero-slack boundary still passes an inclusive delivery requirement; "
                "it is not a strict breaking witness.",
                "",
            ]
        )
    if threshold.status == "unreachable":
        lines.extend(
            [
                "No realization in the declared box reaches this shortfall margin. "
                "This statement is specific to the requested margin.",
                "",
            ]
        )
    if threshold.operation is not None:
        op = threshold.operation
        lines.extend(
            [
                "| Quantity | Value |",
                "| --- | --- |",
                *[
                    f"| {LABELS[name]} | {100 * value:.6f}% |"
                    for name, value in op.realization.items()
                ],
                f"| Maximum delivery | {number(op.maximum_heat_mw)} MW thermal |",
                f"| Required delivery | {number(op.required_heat_mw)} MW thermal |",
                f"| Delivery slack | {number(op.slack_mw)} MW thermal |",
                f"| Electrical input at maximum delivery | {number(op.power_mw)} MW electrical |",
                f"| Source heat at maximum delivery | {number(op.source_heat_mw)} MW thermal |",
                f"| Binding limits | {', '.join(LABELS[name] for name in op.binding_limits) or 'unresolved'} |",
                "",
            ]
        )
    lines.extend(
        [
            "| Failure branch | Search status | Severity | Solve evidence |",
            "| --- | --- | --- | --- |",
            *[
                f"| {LABELS[row.limit]} | {row.status} | "
                f"{'not applicable' if row.status == 'unreachable' else number(row.distance)} | "
                f"{cell(row.evidence.message)} |"
                for row in threshold.branches
            ],
            "",
            "Minimum verification requires both failure branches to be resolved, "
            "LP residual and independent analytical-distance checks to pass, and "
            "the selected realization's recourse evaluation to agree.",
            "",
        ]
    )
    return "\n".join(lines)


def render_storage_audit(audit: StorageAudit) -> str:
    lines = [
        f"## {audit.mode.replace('_', ' ').capitalize()} operation",
        "",
        f"Verdict: **{audit.status}**. Coverage: {audit.coverage}.",
        "",
        "| Future | Minimum prepared energy (MWh) | Maximum prepared energy (MWh) | Event power restrictions |",
        "| --- | --- | --- | --- |",
        *[
            f"| {cell(row.scenario)} | {number(row.minimum_mwh)} | {number(row.maximum_mwh)} | "
            f"{cell('; '.join(row.power_violations)) or 'satisfied'} |"
            for row in audit.intervals
        ],
        "",
        "These intervals include preparation reachability and final-state requirements. "
        "A lower bound above its upper bound means that future is individually infeasible.",
        "",
    ]
    if audit.fixed_preparation_power_mw is not None:
        lines.extend(
            [
                f"Fixed preparation charge: {audit.fixed_preparation_power_mw:g} MW thermal.",
                "",
            ]
        )
    if audit.witness:
        witness = audit.witness
        lines.extend(
            [
                f"Witness: **{witness.kind.replace('_', ' ')}**.",
                "",
                witness.explanation,
                "",
                f"Futures: {', '.join(witness.scenarios)}. "
                f"Minimum cardinality under this operating contract: {len(witness.scenarios)}.",
                "",
            ]
        )
    if audit.dispatch:
        lines.extend(
            [
                "| Future | Preparation information | Prep charge (MW) | Prepared energy (MWh) | Event charge / discharge (MW) | Final energy (MWh) |",
                "| --- | --- | --- | --- | --- | --- |",
                *[
                    f"| {cell(row.scenario)} | {cell(row.preparation_information)} | "
                    f"{number(row.preparation_charge_mw)} | {number(row.prepared_energy_mwh)} | "
                    f"{number(row.event_charge_mw)} / {number(row.event_discharge_mw)} | "
                    f"{number(row.terminal_energy_mwh)} |"
                    for row in audit.dispatch
                ],
                "",
            ]
        )
    lines.extend(
        [
            f"Evidence: {audit.evidence.status}; {audit.evidence.message}.",
            f"Solver: {audit.evidence.solver}. Numerical tolerance: {audit.evidence.tolerance:g}.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class RepairResult(Artifact):
    action: PricedRepair
    audit: ClaimAudit


@dataclass(frozen=True)
class RepairComparison(Artifact):
    candidates: tuple[RepairResult, ...]

    @property
    def selected(self) -> RepairResult | None:
        return next(
            (row for row in self.candidates if row.audit.status == "pass"), None
        )

    def to_markdown(self) -> str:
        lines = [
            "## Repair comparison",
            "",
            "Each replacement design is audited over the full original box, with "
            "the same service requirement and control permissions.",
            "",
            "| Candidate | Added CAPEX (GBP) | Minimum slack (MW thermal) | Verdict |",
            "| --- | --- | --- | --- |",
            *[
                f"| {cell(row.action.change.name)} | {row.action.added_capex_gbp:,.0f} | "
                f"{number(row.audit.worst.slack_mw)} | {row.audit.status} |"
                for row in self.candidates
            ],
            "",
        ]
        if self.selected is None:
            lines.append("No catalog candidate has a verified passing audit.")
        else:
            lines.append(
                f"Cheapest verified passing candidate in this catalog: "
                f"**{self.selected.action.change.name}**, "
                f"£{self.selected.action.added_capex_gbp:,.0f}."
            )
            if (
                abs(self.selected.audit.worst.slack_mw)
                <= self.selected.audit.worst.evidence.tolerance
            ):
                lines.append(
                    "It passes at the boundary, with zero worst-case headroom."
                )
        if any(row.audit.status == "inconclusive" for row in self.candidates):
            lines.append(
                "Unresolved candidates remain in the table and may alter the selection."
            )
        lines.extend(
            [
                "",
                "Costs retain each action's supplied provenance. This comparison does "
                "not establish an optimum outside the catalog or include OPEX.",
                "",
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class StorageRepairResult(Artifact):
    action: PricedRepair
    causal: StorageAudit
    perfect_foresight: StorageAudit


@dataclass(frozen=True)
class StorageRepairComparison(Artifact):
    candidates: tuple[StorageRepairResult, ...]

    @property
    def selected(self) -> StorageRepairResult | None:
        return next(
            (row for row in self.candidates if row.causal.status == "pass"), None
        )

    def to_markdown(self) -> str:
        lines = [
            "## Repair comparison",
            "",
            "| Candidate | Added CAPEX (GBP) | Information change | Causal verdict | Perfect-foresight verdict |",
            "| --- | --- | --- | --- | --- |",
            *[
                f"| {cell(row.action.change.name)} | {row.action.added_capex_gbp:,.0f} | "
                f"{'Exact event identity before preparation' if row.action.change.reveal_before_preparation else 'None'} | "
                f"{row.causal.status} | {row.perfect_foresight.status} |"
                for row in self.candidates
            ],
            "",
        ]
        if self.selected:
            lines.append(
                f"Cheapest verified causal repair in this catalog: **{self.selected.action.change.name}** "
                f"(£{self.selected.action.added_capex_gbp:,.0f})."
            )
        else:
            lines.append("No candidate has a verified passing causal audit.")
        lines.extend(
            [
                "",
                "Every candidate preserves the physical futures and service commitments. "
                "The information intervention explicitly changes when event identity becomes "
                "known; it assumes a correct early signal. It does not assert that an ordinary "
                "forecast reveals the future perfectly. Costs are supplied one-time GBP CAPEX.",
                "",
            ]
        )
        if any(row.causal.status == "inconclusive" for row in self.candidates):
            lines.append(
                "Unresolved candidates remain visible and may change the selection.\n"
            )
        return "\n".join(lines)


def compare_heat_repairs(claim, prices):
    prices = tuple(prices)
    if not all(
        isinstance(p, PricedRepair) and isinstance(p.change, RepairAction)
        for p in prices
    ):
        raise ValueError("expected priced heat repairs")
    prices = sorted(prices, key=lambda p: (p.added_capex_gbp, p.change.name))
    audited = claim.compare_repairs(tuple(p.change for p in prices))
    return RepairComparison(
        tuple(RepairResult(p, r.audit) for p, r in zip(prices, audited.candidates))
    )


def compare_storage_repairs(claim, prices):
    prices = tuple(prices)
    if not all(
        isinstance(p, PricedRepair) and isinstance(p.change, StorageRepairAction)
        for p in prices
    ):
        raise ValueError("expected priced storage repairs")
    prices = sorted(prices, key=lambda p: (p.added_capex_gbp, p.change.name))
    audited = claim.compare_repairs(tuple(p.change for p in prices))
    return StorageRepairComparison(
        tuple(
            StorageRepairResult(p, r.causal, r.perfect_foresight)
            for p, r in zip(prices, audited.candidates)
        )
    )
