"""Engineering evidence for the constant-COP reference model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from .primitives import RepairAction


class Artifact:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)


@dataclass(frozen=True)
class SolveEvidence(Artifact):
    status: str
    message: str
    solver: str
    tolerance: float
    max_residual: float | None = None
    reference_error: float | None = None


@dataclass(frozen=True)
class OperationResult(Artifact):
    status: str
    realization: dict[str, float]
    required_heat_mw: float
    maximum_heat_mw: float | None
    power_mw: float | None
    source_heat_mw: float | None
    slack_mw: float | None
    binding_limits: tuple[str, ...]
    evidence: SolveEvidence
    capacity_lower_mw: float | None = None
    capacity_upper_mw: float | None = None


@dataclass(frozen=True)
class ClaimAudit(Artifact):
    claim_name: str
    status: str
    nominal: OperationResult
    worst: OperationResult
    coverage: str = "full declared box; monotone constant-COP capacity model"
    recourse: str = (
        "fully observed static controls, 0 <= electrical power <= installed capacity"
    )


@dataclass(frozen=True)
class BranchResult(Artifact):
    limit: str
    status: str
    distance: float | None
    realization: dict[str, float] | None
    reference_distance: float | None
    evidence: SolveEvidence


@dataclass(frozen=True)
class ThresholdResult(Artifact):
    kind: str
    status: str
    shortfall_margin_mw: float
    distance: float | None
    minimum_verified: bool
    operation: OperationResult | None
    branches: tuple[BranchResult, ...]
    distance_metric: str = "normalized_linf"


@dataclass(frozen=True)
class RepairResult(Artifact):
    action: RepairAction
    audit: ClaimAudit


@dataclass(frozen=True)
class RepairComparison(Artifact):
    candidates: tuple[RepairResult, ...]
