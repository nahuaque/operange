"""Finite-tree operating plans, conflicting futures, and engineering changes."""

from __future__ import annotations

from dataclasses import dataclass

from ._storage_reference import PreparationInterval
from .results import Artifact, SolveEvidence
from .storage_types import StorageRepairAction


@dataclass(frozen=True)
class StorageDispatch(Artifact):
    scenario: str
    preparation_information: str
    preparation_charge_mw: float
    prepared_energy_mwh: float
    event_charge_mw: float
    event_discharge_mw: float
    terminal_energy_mwh: float


@dataclass(frozen=True)
class StorageWitness(Artifact):
    kind: str
    scenarios: tuple[str, ...]
    preparation_information: str
    minimum_prepared_energy_mwh: float
    maximum_prepared_energy_mwh: float
    gap_mwh: float | None
    explanation: str
    minimum_cardinality: bool = True


@dataclass(frozen=True)
class StorageAudit(Artifact):
    claim_name: str
    mode: str
    status: str
    contract: dict
    intervals: tuple[PreparationInterval, ...]
    dispatch: tuple[StorageDispatch, ...]
    witness: StorageWitness | None
    evidence: SolveEvidence
    fixed_preparation_power_mw: float | None = None
    coverage: str = "complete declared finite two-period scenario tree"


@dataclass(frozen=True)
class StorageRepairResult(Artifact):
    action: StorageRepairAction
    causal: StorageAudit
    perfect_foresight: StorageAudit


@dataclass(frozen=True)
class StorageRepairComparison(Artifact):
    candidates: tuple[StorageRepairResult, ...]
