"""Audited controller synthesis reports, separate from adjustable claim verdicts."""

from dataclasses import dataclass
from functools import cached_property
import json
from typing import ClassVar

from .claim import Claim
from .contract_types import Diagnostic, Evidence, freeze, plain, reference, snapshot
from .controllers import AffineController
from .engineering_results import RobustnessResult


@dataclass(frozen=True)
class ControllerSynthesis:
    """A numerical policy search and the independent audit of its saved commands.

    ``completed`` requires a passing audit and checked objective enclosure.
    ``unresolved`` never establishes infeasibility of the affine policy class or
    of adjustable operation. Solver optimality is not independently certified.
    Use ``freeze()`` to export an accepted candidate through FrozenController.
    """

    source: Claim
    request: dict
    execution: str
    controller: AffineController | None = None
    audit: RobustnessResult | None = None
    evidence: tuple[Evidence, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    schema_version: ClassVar[str] = "controller_synthesis/v1"

    def __post_init__(self):
        object.__setattr__(self, "request", snapshot(self.request))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if self.execution not in ("completed", "unresolved", "unsupported", "invalid"):
            raise ValueError("invalid synthesis execution status")
        if self.audit is not None and (
            self.controller is None
            or self.audit.contract_ref != self.claim.contract.ref
        ):
            raise ValueError(
                "synthesis audit must cover the returned controller binding"
            )
        if self.execution == "completed" and (
            self.audit is None
            or self.audit.execution != "completed"
            or self.audit.payload.verdict != "pass"
            or not any(e.evidence_id == "objective" for e in self.evidence)
        ):
            raise ValueError(
                "completed synthesis requires a passing audit and objective"
            )

    @property
    def verified(self):
        """Whether the candidate passed physical auditing and objective enclosure."""
        return self.execution == "completed"

    @property
    def claim(self):
        """The candidate binding, also available for diagnosing a failed search."""
        if self.controller is None:
            raise ValueError("synthesis did not return a controller candidate")
        return self.source.with_controller(self.controller)

    def freeze(self):
        if not self.verified:
            raise ValueError("only independently verified synthesis can be frozen here")
        return self.claim.freeze()

    def _body(self, compact=False):
        return {
            "schema_version": self.schema_version,
            "source_contract": self.source.contract.to_dict(),
            "request": plain(self.request),
            "execution": self.execution,
            "verified": self.verified,
            "controller": None
            if self.controller is None
            else self.controller.to_manifest(),
            "audit": None
            if self.audit is None
            else self.audit.to_dict(compact=compact),
            "evidence": [e.to_dict() for e in self.evidence],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    @cached_property
    def ref(self):
        return reference(freeze(self._body()), self.schema_version)

    def to_dict(self, *, compact=False):
        return {**self._body(compact), "synthesis_id": self.ref.artifact_id}

    def to_json(self, *, compact=False, indent=2):
        return json.dumps(
            self.to_dict(compact=compact),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )


def synthesize_controller(
    claim, *, objective=None, name="Synthesized affine controller", max_vertices=256
):
    """Choose permitted affine coefficients using the optional CVXPY backend.

    An explicit LinearObjective or ControlTrackingObjective overrides the model's
    objective for this search without changing the model. Complete generators are
    required for a built-in finite set, box or explicit convex hull. There is no
    sampling fallback, clipping, state, or automatic permission expansion.
    """
    from ._controller_synthesis import synthesize

    return synthesize(claim, objective=objective, name=name, max_vertices=max_vertices)
