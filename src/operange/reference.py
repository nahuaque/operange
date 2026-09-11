"""Bounded heat-recovery, storage, startup and heat-cascade reference models.

These model-specific contracts do not imply a general flowsheet or tree solver.
"""

from .heat_recovery import HeatRecoveryClaim
from .adapters import HeatRecoveryAdapter, ThermalStorageAdapter
from .policy import FrozenHeatRecoveryContract
from .primitives import HeatDemand, HeatRecoveryDesign, RepairAction
from .results import ClaimAudit, OperationResult, RepairComparison, ThresholdResult
from .storage import ThermalStorageClaim
from .storage_controllers import StorageController, StoragePreparationRule
from .storage_types import (
    ScenarioTree,
    StorageDesign,
    StorageRepairAction,
    StorageRequirement,
    StorageScenario,
)
from .storage_results import (
    StorageAudit,
    StorageDispatch,
    StorageRepairComparison,
    StorageWitness,
)
from .primitives import BoxSet, Parameter
from .startup import StartupEvent, StartupLoadAdapter
from .pinch import HeatCascadeAdapter, SensibleHeatStream


def example_claim() -> HeatRecoveryClaim:
    return HeatRecoveryClaim(
        name="Meet heat demand throughout the declared operating envelope",
        design=HeatRecoveryDesign(),
        uncertainty=BoxSet(
            tuple(
                Parameter(
                    name=name,
                    unit="fraction",
                    nominal=0,
                    lower=0,
                    upper=0.25,
                    scale=0.25,
                    provenance="Synthetic engineering envelope; not calibrated history",
                )
                for name in ("source_derating", "demand_increase")
            )
        ),
    )


def example_storage_claim() -> ThermalStorageClaim:
    return ThermalStorageClaim(
        name="Meet heat demand and absorb mandatory recovered heat",
        design=StorageDesign(),
        tree=ScenarioTree(
            (
                StorageScenario(
                    "Demand surge",
                    demand_mw=1,
                    recovered_heat_mw=0,
                    provenance="Synthetic one-hour demand event",
                ),
                StorageScenario(
                    "Heat surplus",
                    demand_mw=0,
                    recovered_heat_mw=1,
                    provenance="Synthetic one-hour mandatory heat-recovery event",
                ),
            )
        ),
    )


__all__ = [
    "HeatCascadeAdapter",
    "SensibleHeatStream",
    "StartupEvent",
    "StartupLoadAdapter",
    "HeatRecoveryClaim",
    "HeatRecoveryAdapter",
    "ThermalStorageAdapter",
    "FrozenHeatRecoveryContract",
    "HeatDemand",
    "HeatRecoveryDesign",
    "RepairAction",
    "ClaimAudit",
    "OperationResult",
    "RepairComparison",
    "ThresholdResult",
    "ThermalStorageClaim",
    "StorageController",
    "StoragePreparationRule",
    "ScenarioTree",
    "StorageDesign",
    "StorageRepairAction",
    "StorageRequirement",
    "StorageScenario",
    "StorageAudit",
    "StorageDispatch",
    "StorageRepairComparison",
    "StorageWitness",
    "example_claim",
    "example_storage_claim",
]
