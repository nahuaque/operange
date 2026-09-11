"""Analytical reachability check independent of the storage LP matrices."""

from dataclasses import dataclass

from .storage_types import (
    ScenarioTree,
    StorageDesign,
    StorageRequirement,
    StorageScenario,
)


@dataclass(frozen=True)
class PreparationInterval:
    scenario: str
    minimum_mwh: float
    maximum_mwh: float
    power_violations: tuple[str, ...]

    def feasible(self, tolerance: float) -> bool:
        return (
            not self.power_violations
            and self.minimum_mwh <= self.maximum_mwh + tolerance
        )


def preparation_interval(
    design: StorageDesign,
    requirement: StorageRequirement,
    tree: ScenarioTree,
    scenario: StorageScenario,
    tolerance: float,
) -> PreparationInterval:
    """Eliminate event controls to derive the complete allowable E1 interval.

    Exact heat balance and prohibition of simultaneous charging/discharging
    determine the event action from recovered heat minus demand. The resulting
    final energy is beta*E1 + offset. Intersect its terminal restrictions with
    the independently reachable preparation states.
    """
    alpha = 1 - design.loss_rate_per_hour * tree.preparation_hours
    beta = 1 - design.loss_rate_per_hour * tree.event_hours
    reachable_low = alpha * requirement.initial_energy_mwh
    reachable_high = min(
        design.capacity_mwh,
        reachable_low
        + design.charge_efficiency * design.charging_power_mw * tree.preparation_hours,
    )
    surplus = scenario.recovered_heat_mw - scenario.demand_mw
    charge = max(surplus, 0)
    discharge = max(-surplus, 0)
    offset = (
        design.charge_efficiency * charge - discharge / design.discharge_efficiency
    ) * tree.event_hours
    terminal_high = min(
        design.capacity_mwh,
        requirement.terminal_max_mwh
        if requirement.terminal_max_mwh is not None
        else design.capacity_mwh,
    )
    violations = []
    if charge > design.charging_power_mw + tolerance:
        violations.append("required event charging exceeds installed charging power")
    if discharge > design.discharging_power_mw + tolerance:
        violations.append(
            "required event discharge exceeds installed discharging power"
        )
    return PreparationInterval(
        scenario.name,
        max(reachable_low, (requirement.terminal_min_mwh - offset) / beta),
        min(reachable_high, (terminal_high - offset) / beta),
        tuple(violations),
    )
