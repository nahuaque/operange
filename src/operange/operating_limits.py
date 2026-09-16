"""Portable bounds on a guaranteed operating range under declared operation."""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction as F
from functools import cached_property
import json
from typing import Any, ClassVar, Literal

from .contract_types import Diagnostic, ProcessContract, Record, plain, reference
from .engineering_results import RobustnessResult, result_from_json


@dataclass(frozen=True)
class OperatingLimitTrial(Record):
    """Audit of the original box with one coordinate's upper endpoint changed."""

    endpoint: float
    audit: RobustnessResult

    def _validate(self):
        if self.audit.request.get("query") != "audit" or self.audit.payload.search:
            raise ValueError("operating-limit trials require physical audits")

    @property
    def outcome(self):
        if self.audit.payload.verdict == "pass" and self.audit.execution == "completed":
            return "pass"
        if self.audit.payload.verdict == "fail" and self.audit.payload.witness:
            return "fail"
        return "unresolved"


@dataclass(frozen=True)
class OperatingLimitResult(Record):
    """An achievable endpoint and, when proved, an excluded endpoint.

    Every guarantee covers the entire interval from the declared lower endpoint,
    jointly with all other uncertainty. A passing search ceiling is not a global
    maximum. Unresolved audits never become upper bounds. Loading checks content
    identities; use verify_result() to independently check the mathematics.
    """

    contract: ProcessContract
    request: Mapping[str, Any]
    execution: Literal["completed", "unresolved", "unsupported", "invalid"]
    trials: tuple[OperatingLimitTrial, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    schema_version: ClassVar[str] = "operating_limit/v1"

    def _validate(self):
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError("operating-limit certificates have contradictory bounds")
        if self.execution == "completed" and self.resolution not in (
            "limit_bracketed",
            "ceiling_verified",
            "baseline_infeasible",
        ):
            raise ValueError("completed operating-limit search needs conclusive bounds")
        if self.execution in ("unsupported", "invalid") and self.trials:
            raise ValueError("rejected operating-limit queries cannot contain trials")
        if self.trials:
            if self.parameter is None:
                raise ValueError(
                    "operating-limit trials need a declared scalar coordinate"
                )
            if (
                not isinstance(self.request.get("tolerance"), (int, float))
                or isinstance(self.request["tolerance"], bool)
                or self.request["tolerance"] <= 0
            ):
                raise ValueError("operating-limit trials need a positive tolerance")

    @property
    def contract_ref(self):
        return self.contract.ref

    @property
    def parameter(self):
        name = self.request.get("parameter")
        return next(
            (
                p
                for p in self.contract.domain.get("parameters", ())
                if isinstance(p, Mapping) and p.get("name") == name and "lower" in p
            ),
            None,
        )

    @property
    def unit(self):
        return None if self.parameter is None else self.parameter["unit"]

    @property
    def policy_scope(self):
        if self.contract.operating.get("response") == "frozen_affine_controller":
            return "saved_controller"
        rules = self.contract.operating.get("recourse_policy", {}).get("rules", ())
        if self.contract.model_id == "linear_process/v1" and rules:
            return (
                "fixed_commands"
                if all(r.get("fixed_value") is not None for r in rules)
                else "adjustable_dispatch"
            )
        return "unsupported"

    @property
    def achievable_trial(self):
        return max(
            (t for t in self.trials if t.outcome == "pass"),
            key=lambda t: t.endpoint,
            default=None,
        )

    @property
    def excluded_trial(self):
        return min(
            (t for t in self.trials if t.outcome == "fail"),
            key=lambda t: t.endpoint,
            default=None,
        )

    @property
    def lower(self):
        trial = self.achievable_trial
        return None if trial is None else trial.endpoint

    @property
    def upper(self):
        trial = self.excluded_trial
        return None if trial is None else trial.endpoint

    @property
    def resolution(self):
        if self.execution in ("unsupported", "invalid"):
            return "not_assessed"
        p = self.parameter
        if p is not None:
            if self.upper == p["lower"] and self.lower is None:
                return "baseline_infeasible"
            if self.lower == p["upper"] and self.upper is None:
                return "ceiling_verified"
            if self.lower is not None and self.upper is not None:
                if F(self.upper) - F(self.lower) <= F(self.request["tolerance"]):
                    return "limit_bracketed"
        return "unresolved"

    @property
    def limiting_witness(self):
        trial = self.excluded_trial
        return None if trial is None else trial.audit.payload.witness

    @property
    def limiting_constraints(self):
        witness = self.limiting_witness
        return () if witness is None else witness.affected_constraints

    def as_claim(self):
        """Reconstruct the checked achievable range, retaining the saved policy.

        Distance metrics are not part of this query; attach one explicitly if
        distance searches are needed on the reconstructed claim.
        """
        if self.achievable_trial is None:
            raise ValueError("no guaranteed operating range was established")
        from .controllers import AffineController
        from .domain_io import domain_from_manifest
        from .linear_process import LinearProcessAdapter
        from .recourse import RecoursePolicy

        contract = self.achievable_trial.audit.contract
        controller = contract.operating.get("controller")
        model = LinearProcessAdapter(**plain(contract.model))
        claim = model.as_claim(
            domain_from_manifest(plain(contract.domain)),
            recourse=RecoursePolicy(**plain(contract.operating["recourse_policy"])),
            requirements=contract.operating["requirement_refs"],
            controller=None
            if controller is None
            else AffineController.from_json(json.dumps(plain(controller))),
        )
        if claim.contract != contract:
            raise ValueError("operating-limit replay changed the checked contract")
        return claim

    def _body(self, compact=False):
        return {
            "schema_version": self.schema_version,
            "contract": self.contract.to_dict(),
            "request": plain(self.request),
            "execution": self.execution,
            "resolution": self.resolution,
            "policy_scope": self.policy_scope,
            "unit": self.unit,
            "lower": self.lower,
            "upper": self.upper,
            "trials": [
                {"endpoint": t.endpoint, "audit": t.audit.to_dict(compact=compact)}
                for t in self.trials
            ],
            "diagnostics": plain(self.diagnostics),
        }

    @cached_property
    def ref(self):
        return reference(self._body(), self.schema_version)

    @property
    def result_id(self):
        return self.ref.artifact_id

    def to_dict(self, *, compact=False):
        return {**self._body(compact), "result_id": self.result_id}

    def to_json(self, *, compact=False, indent=2):
        return json.dumps(
            self.to_dict(compact=compact),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, document):
        try:
            return cls._load_json(document)
        except (
            TypeError,
            KeyError,
            AttributeError,
            IndexError,
            StopIteration,
            OverflowError,
        ) as exc:
            raise ValueError(f"invalid operating-limit artifact: {exc}") from exc

    @classmethod
    def _load_json(cls, document):
        from .controllers import _fields, _read_json

        data = _read_json(document)
        _fields(
            data,
            (
                "schema_version",
                "contract",
                "request",
                "execution",
                "resolution",
                "policy_scope",
                "unit",
                "lower",
                "upper",
                "trials",
                "diagnostics",
                "result_id",
            ),
        )
        if data["schema_version"] != cls.schema_version:
            raise ValueError("unsupported operating-limit schema")
        if not isinstance(data["trials"], list) or not isinstance(
            data["diagnostics"], list
        ):
            raise ValueError("operating-limit trials and diagnostics must be arrays")
        trials = []
        for t in data["trials"]:
            _fields(t, ("endpoint", "audit"))
            trials.append(
                OperatingLimitTrial(
                    t["endpoint"], result_from_json(json.dumps(t["audit"]))
                )
            )
        result = cls(
            data["contract"],
            data["request"],
            data["execution"],
            tuple(trials),
            tuple(data["diagnostics"]),
        )
        for name in (
            "resolution",
            "policy_scope",
            "unit",
            "lower",
            "upper",
            "result_id",
        ):
            if data[name] != getattr(result, name):
                raise ValueError("operating-limit identity or derived bounds differ")
        return result


def operating_limit(
    claim,
    parameter,
    *,
    tolerance=1e-6,
    max_iterations=64,
    max_vertices=256,
    backend="scipy",
):
    """Find a guaranteed prefix of one scalar coordinate in a declared box.

    The parameter's original lower/upper bounds declare the operating floor and
    search ceiling. Other uncertainty, physical limits, and operation stay fixed.
    Supported models are LinearProcessAdapter with fixed/fully observed commands,
    or a saved affine controller (including uncertain measurements).
    """
    from ._operating_limit import search

    return search(
        claim,
        parameter,
        tolerance=tolerance,
        max_iterations=max_iterations,
        max_vertices=max_vertices,
        backend=backend,
    )
