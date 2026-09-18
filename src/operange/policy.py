"""Portable freezing for the explicit, constant-COP prototype only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from ._artifact_json import (
    check_identity,
    digest,
    dumps,
    read_json,
    require_fields,
    require_schema,
)
from .heat_recovery import MODEL_ID, HeatRecoveryClaim
from .primitives import BoxSet, HeatDemand, HeatRecoveryDesign, Parameter
from .results import OperationResult


@dataclass(frozen=True)
class FrozenHeatRecoveryContract:
    """Fixed design/claim, engineering scales, and numerical evaluation policy.

    Replays fully observed static recourse. This is not a causal controller or
    a deployment artifact. The fingerprint identifies content, not authorship.
    """

    claim: HeatRecoveryClaim

    def __post_init__(self) -> None:
        if not isinstance(self.claim, HeatRecoveryClaim):
            raise ValueError("claim must be a HeatRecoveryClaim")

    def _payload(self) -> dict:
        claim = asdict(self.claim)
        # Coordinate order is semantically irrelevant; normalize the manifest.
        claim["uncertainty"]["parameters"] = sorted(
            claim["uncertainty"]["parameters"], key=lambda parameter: parameter["name"]
        )
        return {"schema_version": 1, "model_id": MODEL_ID, "claim": claim}

    @property
    def fingerprint(self) -> str:
        return digest(self._payload())

    def to_json(self) -> str:
        return dumps(
            {**self._payload(), "fingerprint": self.fingerprint},
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, document: str) -> FrozenHeatRecoveryContract:
        return cls.from_dict(read_json(document))

    @classmethod
    def from_dict(cls, payload):
        require_fields(payload, {"schema_version", "model_id", "claim", "fingerprint"})
        require_schema(payload, 1)
        if payload["model_id"] != MODEL_ID:
            raise ValueError("unsupported model_id")
        claim = payload["claim"]
        require_fields(
            claim,
            {
                "name",
                "design",
                "uncertainty",
                "requirement",
                "cop",
                "recourse",
                "tolerance",
            },
        )
        require_fields(claim["design"], {"source_capacity_mw", "power_capacity_mw"})
        require_fields(claim["requirement"], {"base_heat_mw", "reserve_mw"})
        require_fields(claim["uncertainty"], {"parameters"})
        parameters = claim["uncertainty"]["parameters"]
        if not isinstance(parameters, list):
            raise ValueError("parameters must be an array")
        for parameter in parameters:
            require_fields(
                parameter,
                {"name", "unit", "nominal", "lower", "upper", "scale", "provenance"},
            )
        restored = cls(
            HeatRecoveryClaim(
                name=claim["name"],
                design=HeatRecoveryDesign(**claim["design"]),
                uncertainty=BoxSet(tuple(Parameter(**p) for p in parameters)),
                requirement=HeatDemand(**claim["requirement"]),
                cop=claim["cop"],
                recourse=claim["recourse"],
                tolerance=claim["tolerance"],
            )
        )
        check_identity(payload["fingerprint"], restored.fingerprint)
        return restored

    def audit_realization(self, realization: Mapping[str, float]) -> OperationResult:
        return self.claim.evaluate(realization)
