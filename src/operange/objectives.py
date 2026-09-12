"""Explicit operating preferences, separate from physical feasibility limits."""

from dataclasses import dataclass
from typing import Literal

from .contract_types import Record, nonempty, unique


@dataclass(frozen=True)
class LinearObjective(Record):
    """Optimize one declared affine output, in that output's physical unit."""

    output: str
    sense: Literal["minimize", "maximize"] = "minimize"
    tolerance: float = 1e-8
    provenance: str = "Caller-declared operating preference"

    def _validate(self):
        nonempty(self.output, "objective output")
        nonempty(self.provenance, "objective provenance")
        if self.tolerance < 0:
            raise ValueError("objective tolerance must be nonnegative")


@dataclass(frozen=True)
class ControlTarget(Record):
    """A physical target and scale for weight * ((control - target)/scale)**2."""

    control: str
    target: float
    scale: float
    unit: str
    weight: float = 1.0

    def _validate(self):
        nonempty(self.control, "target control")
        nonempty(self.unit, "target unit")
        if self.scale <= 0 or self.weight <= 0:
            raise ValueError("target scales and weights must be positive")


@dataclass(frozen=True)
class ControlTrackingObjective(Record):
    """Minimize a dimensionless sum of weighted squared target departures.

    Targeting every adjustable control makes the objective strictly convex.
    Targets express preferences and need not satisfy equipment or service limits.
    """

    targets: tuple[ControlTarget, ...]
    quantity_id: str = "dispatch_tracking_error"
    tolerance: float = 1e-8
    provenance: str = "Caller-declared command targets"

    def _validate(self):
        if not self.targets:
            raise ValueError("tracking objective needs at least one control target")
        unique(tuple(t.control for t in self.targets), "target controls")
        nonempty(self.quantity_id, "tracking objective quantity")
        nonempty(self.provenance, "objective provenance")
        if self.tolerance < 0:
            raise ValueError("objective tolerance must be nonnegative")
