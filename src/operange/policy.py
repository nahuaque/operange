"""Portable freezing for the explicit, constant-COP prototype only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

from .heat_recovery import MODEL_ID, HeatRecoveryClaim
from .primitives import BoxSet, HeatDemand, HeatRecoveryDesign, Parameter
from .results import OperationResult


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _keys(payload: dict, expected: set[str]) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"expected exactly these fields: {sorted(expected)}")


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
        return hashlib.sha256(_canonical(self._payload()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return json.dumps(
            {**self._payload(), "fingerprint": self.fingerprint},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, document: str) -> FrozenHeatRecoveryContract:
        payload = json.loads(document, object_pairs_hook=_unique)
        _keys(payload, {"schema_version", "model_id", "claim", "fingerprint"})
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported contract schema_version")
        if payload["model_id"] != MODEL_ID:
            raise ValueError("unsupported model_id")
        claim = payload["claim"]
        _keys(
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
        _keys(claim["design"], {"source_capacity_mw", "power_capacity_mw"})
        _keys(claim["requirement"], {"base_heat_mw", "reserve_mw"})
        _keys(claim["uncertainty"], {"parameters"})
        parameters = claim["uncertainty"]["parameters"]
        if not isinstance(parameters, list):
            raise ValueError("parameters must be an array")
        for parameter in parameters:
            _keys(
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
        if payload["fingerprint"] != restored.fingerprint:
            raise ValueError("contract fingerprint mismatch")
        return restored

    def audit_realization(self, realization: Mapping[str, float]) -> OperationResult:
        return self.claim.evaluate(realization)
