"""Causal storage decisions on a finite, fully specified two-period tree."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

from ._storage_reference import PreparationInterval, preparation_interval
from .linear import solve_lp
from .primitives import finite
from .storage_results import (
    StorageAudit,
    StorageDispatch,
    StorageRepairComparison,
    StorageRepairResult,
    StorageWitness,
)
from .storage_types import (
    ScenarioTree,
    StorageDesign,
    StorageRepairAction,
    StorageRequirement,
    text_field,
)

STORAGE_MODEL_ID = "thermal_storage_two_period_v1"


@dataclass(frozen=True)
class ThermalStorageClaim:
    """Meet event demand and absorb mandatory recovered heat without dumping.

    Preparation can charge from an available external source. At the event,
    recovered heat plus storage discharge must exactly meet demand plus storage
    charging. Simultaneous charging/discharging is prohibited. Period lengths,
    heat losses, efficiencies, and initial/final energy are explicit.
    """

    name: str
    design: StorageDesign
    tree: ScenarioTree
    requirement: StorageRequirement = field(default_factory=StorageRequirement)
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        text_field(self.name, "name")
        for name, expected in (
            ("design", StorageDesign),
            ("tree", ScenarioTree),
            ("requirement", StorageRequirement),
        ):
            if not isinstance(getattr(self, name), expected):
                raise ValueError(f"{name} must be a {expected.__name__}")
        object.__setattr__(self, "tolerance", finite(self.tolerance, "tolerance"))
        if not 1e-10 <= self.tolerance <= 1e-4:
            raise ValueError("tolerance must lie between 1e-10 and 1e-4")
        if self.requirement.initial_energy_mwh > self.design.capacity_mwh:
            raise ValueError("initial energy exceeds installed storage capacity")
        for hours in (self.tree.preparation_hours, self.tree.event_hours):
            if self.design.loss_rate_per_hour * hours >= 1:
                raise ValueError("loss_rate_per_hour * period hours must be below one")
            finite(
                self.design.charging_power_mw * self.design.charge_efficiency * hours,
                "charging energy",
            )
            finite(
                self.design.discharging_power_mw
                * hours
                / self.design.discharge_efficiency,
                "discharging energy",
            )
        for scenario in self.tree.scenarios:
            interval = preparation_interval(
                self.design, self.requirement, self.tree, scenario, self.tolerance
            )
            finite(interval.minimum_mwh, "minimum prepared energy")
            finite(interval.maximum_mwh, "maximum prepared energy")

    def _groups(self, mode: str, fixed_preparation_power_mw=None) -> dict[str, str]:
        policy = self.recourse_policy(
            mode, fixed_preparation_power_mw=fixed_preparation_power_mw
        )
        rule = policy.rule("preparation_charge")
        groups = {}
        for scenario in self.tree.scenarios:
            info = rule.information(
                {
                    "preparation_observation": scenario.preparation_observation,
                    "event_identity": scenario.name,
                }
            )
            groups[scenario.name] = (
                "fixed policy"
                if rule.fixed_value is not None
                else f"event known: {info['event_identity']}"
                if "event_identity" in info
                else info["preparation_observation"]
            )
        return groups

    def audit(
        self, mode: str = "causal", *, fixed_preparation_power_mw: float | None = None
    ) -> StorageAudit:
        policy = self.recourse_policy(
            mode, fixed_preparation_power_mw=fixed_preparation_power_mw
        )
        fixed = policy.rule("preparation_charge").fixed_value
        groups = self._groups(mode, fixed)
        intervals = tuple(
            preparation_interval(
                self.design, self.requirement, self.tree, scenario, self.tolerance
            )
            for scenario in sorted(self.tree.scenarios, key=lambda s: s.name)
        )
        reference_witness = self._conflict(intervals, groups, mode, fixed)
        bounds, equalities, rhs, inequalities, upper, group_vars, event_vars = (
            self._compile(groups, fixed)
        )
        objective = [0.0] * len(bounds)
        for charge, _ in group_vars.values():
            objective[charge] = 1.0
        x, evidence = solve_lp(
            objective,
            bounds,
            self.tolerance,
            equalities=equalities,
            rhs=rhs,
            inequalities=inequalities,
            upper=upper,
        )
        dispatch = ()
        witness = None
        status = "inconclusive"
        if (
            x is None
            and evidence.status == "infeasible"
            and reference_witness is not None
        ):
            status, witness = "fail", reference_witness
            evidence = replace(
                evidence,
                message="LP infeasibility agrees with the analytical preparation-state certificate",
            )
        elif x is not None and reference_witness is None:
            initial = self._retained_initial()
            gain = self.design.charge_efficiency * self.tree.preparation_hours
            expected = sum(
                fixed
                if fixed is not None
                else max(
                    0.0,
                    max(
                        row.minimum_mwh
                        for row in intervals
                        if groups[row.scenario] == group
                    )
                    - initial,
                )
                / gain
                for group in group_vars
            )
            error = abs(sum(x[charge] for charge, _ in group_vars.values()) - expected)
            evidence = replace(evidence, reference_error=error)
            if error <= self.tolerance:
                status = "pass"
                evidence = replace(
                    evidence,
                    message="LP plan passes residual checks and independent interval/charging-optimum checks",
                )
                dispatch = tuple(
                    StorageDispatch(
                        scenario.name,
                        groups[scenario.name],
                        x[group_vars[groups[scenario.name]][0]],
                        x[group_vars[groups[scenario.name]][1]],
                        x[event_vars[scenario.name][0]],
                        x[event_vars[scenario.name][1]],
                        x[event_vars[scenario.name][2]],
                    )
                    for scenario in sorted(self.tree.scenarios, key=lambda s: s.name)
                )
        if status == "inconclusive":
            evidence = replace(
                evidence,
                status="unresolved",
                message=f"storage feasibility unresolved or disagrees with analytical check: {evidence.message}",
            )
        return StorageAudit(
            self.name,
            mode,
            status,
            {"model_id": STORAGE_MODEL_ID, **asdict(self)},
            intervals,
            dispatch,
            witness,
            evidence,
            fixed,
        )

    def _retained_initial(self) -> float:
        return (
            1 - self.design.loss_rate_per_hour * self.tree.preparation_hours
        ) * self.requirement.initial_energy_mwh

    def _conflict(
        self,
        intervals: tuple[PreparationInterval, ...],
        groups: dict[str, str],
        mode: str,
        fixed: float | None,
    ) -> StorageWitness | None:
        for row in intervals:
            if not row.feasible(self.tolerance):
                explanation = "; ".join(row.power_violations) or (
                    f"{row.scenario} requires prepared energy at least {row.minimum_mwh:g} MWh "
                    f"and at most {row.maximum_mwh:g} MWh, even with advance knowledge."
                )
                return StorageWitness(
                    "individual_infeasibility",
                    (row.scenario,),
                    groups[row.scenario],
                    row.minimum_mwh,
                    row.maximum_mwh,
                    None if row.power_violations else row.minimum_mwh - row.maximum_mwh,
                    explanation,
                )
        if mode == "fixed":
            prepared = (
                self._retained_initial()
                + self.design.charge_efficiency * fixed * self.tree.preparation_hours
            )
            for row in intervals:
                if (
                    prepared < row.minimum_mwh - self.tolerance
                    or prepared > row.maximum_mwh + self.tolerance
                ):
                    return StorageWitness(
                        "fixed_policy_failure",
                        (row.scenario,),
                        groups[row.scenario],
                        row.minimum_mwh,
                        row.maximum_mwh,
                        max(row.minimum_mwh - prepared, prepared - row.maximum_mwh),
                        f"The fixed preparation rule leaves {prepared:g} MWh. {row.scenario} "
                        f"requires [{row.minimum_mwh:g}, {row.maximum_mwh:g}] MWh. "
                        "This refutes the fixed rule, not every adjustable policy.",
                    )
            return None
        for group in sorted(set(groups.values())):
            rows = [row for row in intervals if groups[row.scenario] == group]
            lower = max(rows, key=lambda row: row.minimum_mwh)
            upper = min(rows, key=lambda row: row.maximum_mwh)
            if lower.minimum_mwh > upper.maximum_mwh + self.tolerance:
                return StorageWitness(
                    "conflicting_futures",
                    (lower.scenario, upper.scenario),
                    group,
                    lower.minimum_mwh,
                    upper.maximum_mwh,
                    lower.minimum_mwh - upper.maximum_mwh,
                    f"Under preparation observation '{group}', {lower.scenario} needs at least "
                    f"{lower.minimum_mwh:g} MWh stored, while {upper.scenario} permits at most "
                    f"{upper.maximum_mwh:g} MWh. Both futures are individually feasible, but "
                    "no shared preparation decision satisfies both. The two-future witness "
                    "is minimal because either future alone admits an operating plan.",
                )
        return None

    def _compile(self, groups: dict[str, str], fixed: float | None):
        """Compile one preparation decision/state per observable history.

        Event balances prohibit dumping and unmet demand. Separate signed-mode
        bounds rule out simultaneous charge/discharge, retaining a linear model.
        """
        d, tree, requirement = self.design, self.tree, self.requirement
        bounds = []
        group_vars, event_vars = {}, {}
        for group in sorted(set(groups.values())):
            group_vars[group] = (len(bounds), len(bounds) + 1)
            bounds.extend(
                [
                    (fixed, fixed) if fixed is not None else (0, d.charging_power_mw),
                    (0, d.capacity_mwh),
                ]
            )
        for scenario in sorted(tree.scenarios, key=lambda s: s.name):
            event_vars[scenario.name] = (len(bounds), len(bounds) + 1, len(bounds) + 2)
            bounds.extend(
                [
                    (
                        0,
                        d.charging_power_mw
                        if scenario.recovered_heat_mw >= scenario.demand_mw
                        else 0,
                    ),
                    (
                        0,
                        d.discharging_power_mw
                        if scenario.demand_mw > scenario.recovered_heat_mw
                        else 0,
                    ),
                    (0, d.capacity_mwh),
                ]
            )
        equalities, rhs, inequalities, upper = [], [], [], []

        def row(entries):
            result = [0.0] * len(bounds)
            for index, coefficient in entries.items():
                result[index] = coefficient
            return result

        for charge, energy in group_vars.values():
            equalities.append(
                row({energy: 1, charge: -d.charge_efficiency * tree.preparation_hours})
            )
            rhs.append(self._retained_initial())
        for scenario in sorted(tree.scenarios, key=lambda s: s.name):
            charge, discharge, final = event_vars[scenario.name]
            _, prepared = group_vars[groups[scenario.name]]
            equalities.append(row({charge: 1, discharge: -1}))
            rhs.append(scenario.recovered_heat_mw - scenario.demand_mw)
            equalities.append(
                row(
                    {
                        final: 1,
                        prepared: -(1 - d.loss_rate_per_hour * tree.event_hours),
                        charge: -d.charge_efficiency * tree.event_hours,
                        discharge: tree.event_hours / d.discharge_efficiency,
                    }
                )
            )
            rhs.append(0)
            inequalities.append(row({final: -1}))
            upper.append(-requirement.terminal_min_mwh)
            if requirement.terminal_max_mwh is not None:
                inequalities.append(row({final: 1}))
                upper.append(requirement.terminal_max_mwh)
        return bounds, equalities, rhs, inequalities, upper, group_vars, event_vars

    def evaluate_result(self, mode="causal", *, fixed_preparation_power_mw=None):
        """Evaluate joint model-and-service feasibility on the declared tree."""
        from .storage_contract import evaluate_result

        try:
            claim = self.as_claim(
                mode, fixed_preparation_power_mw=fixed_preparation_power_mw
            )
        except (ValueError, TypeError):
            return evaluate_result(
                self, mode, fixed_preparation_power_mw=fixed_preparation_power_mw
            )
        return claim.evaluate_result()

    def audit_result(self, mode="causal", *, fixed_preparation_power_mw=None):
        """Return finite-tree robustness and a validated conflict when available."""
        from .storage_contract import audit_result

        try:
            claim = self.as_claim(
                mode, fixed_preparation_power_mw=fixed_preparation_power_mw
            )
        except (ValueError, TypeError):
            return audit_result(
                self, mode, fixed_preparation_power_mw=fixed_preparation_power_mw
            )
        return claim.audit_result()

    def sensitivity_result(self, **request):
        """Return explicit unsupported evidence for the current derivative scope."""
        return self.as_claim().sensitivity_result(**request)

    def recourse_policy(self, mode="causal", *, fixed_preparation_power_mw=None):
        from .recourse import DecisionRule, RecoursePolicy

        if mode not in ("fixed", "causal", "perfect_foresight"):
            raise ValueError("mode must be fixed, causal, or perfect_foresight")
        fixed = fixed_preparation_power_mw
        if mode == "fixed":
            fixed = finite(fixed, "fixed_preparation_power_mw")
            if not 0 <= fixed <= self.design.charging_power_mw:
                raise ValueError(
                    "fixed preparation power must lie within installed limits"
                )
        elif fixed is not None:
            raise ValueError(
                "fixed_preparation_power_mw is only valid for fixed operation"
            )
        observes = (
            ()
            if mode == "fixed"
            else ("event_identity",)
            if mode == "perfect_foresight"
            else ("preparation_observation",)
        )
        return RecoursePolicy(
            mode,
            (
                DecisionRule(
                    "preparation_charge", "MW", "preparation", observes, fixed
                ),
                *(
                    DecisionRule(
                        control,
                        "MW",
                        "event",
                        ("preparation_observation", "event_identity"),
                    )
                    for control in ("event_charge", "event_discharge")
                ),
            ),
        )

    def as_claim(self, mode="causal", *, fixed_preparation_power_mw=None):
        """Bind this tree model to a generic claim with explicit permissions."""
        from .adapters import ThermalStorageAdapter
        from .claim import Claim

        return Claim(
            ThermalStorageAdapter(self),
            self.tree.domain,
            self.recourse_policy(
                mode, fixed_preparation_power_mw=fixed_preparation_power_mw
            ),
        )

    def with_repair(self, action: StorageRepairAction) -> ThermalStorageClaim:
        if not isinstance(action, StorageRepairAction):
            raise ValueError("action must be a StorageRepairAction")
        tree = self.tree
        if action.reveal_before_preparation:
            tree = replace(
                tree,
                scenarios=tuple(
                    replace(s, preparation_observation=f"event identity: {s.name}")
                    for s in tree.scenarios
                ),
            )
        return replace(self, design=action.design or self.design, tree=tree)

    def compare_repairs(
        self, actions: Sequence[StorageRepairAction]
    ) -> StorageRepairComparison:
        """Audit engineering changes in caller order without ranking or selection."""
        actions = tuple(actions)
        if not all(isinstance(action, StorageRepairAction) for action in actions):
            raise ValueError("all candidates must be StorageRepairAction objects")
        if len({action.name for action in actions}) != len(actions):
            raise ValueError("repair names must be unique")
        return StorageRepairComparison(
            tuple(
                StorageRepairResult(
                    action,
                    self.with_repair(action).audit("causal"),
                    self.with_repair(action).audit("perfect_foresight"),
                )
                for action in actions
            )
        )
