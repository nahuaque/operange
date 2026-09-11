"""Re-audit caller-supplied changes against a common engineering commitment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
import json
from typing import Any, ClassVar, Literal

from ._change_contracts import ContractComparison, FieldChange, compare_contracts
from .claim import Claim, rejected_result
from .contract_types import (
    ArtifactRef,
    Record,
    nonempty,
    plain,
    reference,
    unique,
)
from .engineering_results import RobustnessResult, result_from_json


@dataclass(frozen=True)
class EngineeringChange:
    """A named, attributed candidate claim; no mutation of the baseline is implied."""

    name: str
    claim: Claim
    provenance: str

    def __post_init__(self):
        for name in ("name", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            nonempty(value, name)
        if not isinstance(self.claim, Claim):
            raise ValueError("engineering change needs a common Claim; use as_claim()")


@dataclass(frozen=True)
class RequirementSummary(Record):
    """Physical margins in stored responses, not global worst-case bounds.

    For a signed inequality residual r <= 0, margin is -r. For an equality,
    margin is -abs(r). The minimum covers stored responses and array coordinates.
    Tolerance is reported separately and is not added to this physical margin.
    """

    constraint_ref: str
    unit: str
    relation: Literal["eq", "le"]
    tolerance: float
    observed_margin: float | None
    satisfied: int
    violated: int
    unknown: int
    not_evaluated: int
    evaluation_refs: tuple[ArtifactRef, ...]
    scope: Literal["stored_responses"] = "stored_responses"


def _flatten(value):
    if isinstance(value, tuple):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _requirements(audit):
    selected = audit.contract.operating.get("requirement_refs")
    summaries = []
    for spec in audit.contract.constraints:
        if spec.kind != "requirement" or (
            selected is not None and spec.constraint_id not in selected
        ):
            continue
        counts = dict.fromkeys(("satisfied", "violated", "unknown", "not_evaluated"), 0)
        margins, refs = [], []
        for evaluation in audit.supporting_evaluations:
            check = next(
                (
                    c
                    for c in evaluation.payload.constraint_checks
                    if c.constraint_ref == spec.constraint_id
                ),
                None,
            )
            counts[check.assessment if check else "not_evaluated"] += 1
            if check and check.residual is not None:
                margins.extend(
                    -r if spec.relation == "le" else -abs(r)
                    for r in _flatten(check.residual.value)
                )
                refs.append(evaluation.ref)
        summaries.append(
            RequirementSummary(
                spec.constraint_id,
                audit.contract.quantity(spec.residual_ref).unit,
                spec.relation,
                spec.tolerance,
                min(margins) if margins else None,
                **counts,
                evaluation_refs=tuple(refs),
            )
        )
    return tuple(summaries)


def _validate_audit(audit):
    if audit.request.get("query") != "audit" or audit.payload.search is not None:
        raise ValueError("change comparison requires an audit, not a distance search")


def _verdict(audit):
    # A partial audit can prove failure through a verified witness. A pass must
    # finish its advertised audit; unresolved execution cannot restore service.
    if audit.payload.verdict == "fail":
        return "fail"
    if audit.payload.verdict == "pass" and audit.execution == "completed":
        return "pass"
    return None


@dataclass(frozen=True)
class ChangeResult(Record):
    name: str
    provenance: str
    baseline: RobustnessResult
    audit: RobustnessResult
    baseline_distance: Mapping[str, Any] | None = None
    distance: Mapping[str, Any] | None = None

    def _validate(self):
        nonempty(self.name, "change name")
        nonempty(self.provenance, "change provenance")
        _validate_audit(self.baseline)
        _validate_audit(self.audit)

    @cached_property
    def delta(self) -> ContractComparison:
        return compare_contracts(
            self.baseline.contract,
            self.audit.contract,
            self.baseline_distance,
            self.distance,
        )

    @property
    def transition(self):
        if self.delta.commitment != "preserved":
            return "not_comparable"
        before, after = _verdict(self.baseline), _verdict(self.audit)
        if before is None or after is None:
            return "inconclusive"
        return {
            ("fail", "pass"): "restored",
            ("pass", "fail"): "regressed",
            ("fail", "fail"): "still_failing",
            ("pass", "pass"): "still_passing",
        }[before, after]

    @cached_property
    def requirements(self) -> tuple[RequirementSummary, ...]:
        return _requirements(self.audit)

    def _metadata(self):
        return {
            "name": self.name,
            "provenance": self.provenance,
            "audit_ref": plain(self.audit.ref),
            "distance": plain(self.distance),
            "delta": self.delta.to_dict(),
            "transition": self.transition,
            "requirements": plain(self.requirements),
        }


@dataclass(frozen=True)
class ChangeComparison(Record):
    """Portable comparison with complete audits and no economic ranking.

    Compact export deduplicates within each embedded audit. Both encodings have
    the same comparison identity. Loading validates evidence and derived labels;
    it never executes an adapter or re-establishes physical truth.
    """

    baseline: RobustnessResult
    candidates: tuple[ChangeResult, ...]
    baseline_distance: Mapping[str, Any] | None = None
    schema_version: ClassVar[str] = "engineering_change_comparison/v1"

    def _validate(self):
        _validate_audit(self.baseline)
        unique(tuple(c.name for c in self.candidates), "change names")
        for candidate in self.candidates:
            if (
                candidate.baseline.ref != self.baseline.ref
                or candidate.baseline_distance != self.baseline_distance
            ):
                raise ValueError("candidate uses a different baseline")

    @cached_property
    def requirements(self) -> tuple[RequirementSummary, ...]:
        return _requirements(self.baseline)

    def _metadata(self):
        return {
            "schema_version": self.schema_version,
            "baseline_ref": plain(self.baseline.ref),
            "baseline_distance": plain(self.baseline_distance),
            "requirements": plain(self.requirements),
            "candidates": [c._metadata() for c in self.candidates],
        }

    @cached_property
    def ref(self):
        return reference(self._metadata(), self.schema_version)

    @property
    def comparison_id(self):
        return self.ref.artifact_id

    def to_dict(self, *, compact=False):
        audits = (self.baseline, *(c.audit for c in self.candidates))
        return {
            **self._metadata(),
            "comparison_id": self.comparison_id,
            "artifacts": {
                "audits": {a.result_id: a.to_dict(compact=compact) for a in audits}
            },
        }

    def to_json(self, *, indent=2, compact=False):
        return json.dumps(
            self.to_dict(compact=compact),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, text):
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def nonfinite(value):
            raise ValueError(f"nonfinite JSON value: {value}")

        try:
            data = json.loads(
                text, object_pairs_hook=unique_object, parse_constant=nonfinite
            )
            if data["schema_version"] != cls.schema_version:
                raise ValueError("unsupported change comparison schema")
            if set(data["artifacts"]) != {"audits"}:
                raise ValueError("invalid comparison artifact registry")
            audits = {}
            for key, value in data["artifacts"]["audits"].items():
                audit = result_from_json(json.dumps(value, allow_nan=False))
                if not isinstance(audit, RobustnessResult) or key != audit.result_id:
                    raise ValueError("invalid audit identity")
                _validate_audit(audit)
                audits[key] = audit

            def resolve(raw):
                ref = ArtifactRef(**raw)
                audit = audits[ref.artifact_id]
                if audit.ref != ref:
                    raise ValueError("invalid audit reference")
                return audit

            baseline = resolve(data["baseline_ref"])
            candidates = tuple(
                ChangeResult(
                    c["name"],
                    c["provenance"],
                    baseline,
                    resolve(c["audit_ref"]),
                    data["baseline_distance"],
                    c["distance"],
                )
                for c in data["candidates"]
            )
            comparison = cls(baseline, candidates, data["baseline_distance"])
            expected = {
                **comparison._metadata(),
                "comparison_id": comparison.comparison_id,
            }
            actual = {k: v for k, v in data.items() if k != "artifacts"}

            # Canonical JSON comparison also distinguishes bool from numeric 1.
            def canonical(value):
                return json.dumps(value, sort_keys=True, allow_nan=False)

            if canonical(expected) != canonical(actual):
                raise ValueError(
                    "comparison metadata or identity does not match audits"
                )
            used = {baseline.result_id, *(c.audit.result_id for c in candidates)}
            if set(audits) != used:
                raise ValueError("unused audit artifacts")
            return comparison
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("invalid change comparison fields or references") from exc


def _audit(claim):
    contract = claim.contract
    try:
        audit = claim.audit_result()
        _validate_audit(audit)
        return audit
    except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
        return rejected_result(
            contract,
            "audit",
            {"query": "audit"},
            str(exc),
            code="comparison_audit_error",
            execution="unresolved",
        )


def compare_changes(baseline: Claim, changes) -> ChangeComparison:
    """Audit the baseline once and each named candidate once, in caller order.

    All candidates are validated before executing any audit. Adapter failures
    retain an unresolved row and diagnostic, rather than implying infeasibility.
    No equivalence of different domains, service levels or arbitrary models is
    inferred. Known built-in declarations distinguish equipment from service.
    """
    if not isinstance(baseline, Claim):
        raise ValueError("baseline must be a common Claim; use as_claim()")
    changes = tuple(changes)
    if any(not isinstance(change, EngineeringChange) for change in changes):
        raise ValueError("changes must contain EngineeringChange records")
    unique(tuple(change.name for change in changes), "change names")
    # Invalid manifests have no auditable contract and fail before any execution.
    for claim in (baseline, *(c.claim for c in changes)):
        claim.contract
    distance = baseline.distance.to_manifest() if baseline.distance else None
    original = _audit(baseline)
    candidates = tuple(
        ChangeResult(
            c.name,
            c.provenance,
            original,
            _audit(c.claim),
            distance,
            c.claim.distance.to_manifest() if c.claim.distance else None,
        )
        for c in changes
    )
    return ChangeComparison(original, candidates, distance)


__all__ = [
    "EngineeringChange",
    "ChangeResult",
    "ChangeComparison",
    "RequirementSummary",
    "ContractComparison",
    "FieldChange",
    "compare_changes",
]
