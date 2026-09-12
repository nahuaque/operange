"""Independent, solver-free verification of exported engineering certificates."""

from dataclasses import dataclass
import json
from typing import ClassVar, Literal

from .contract_types import ArtifactRef, Record
from .engineering_results import (
    EvaluationResult,
    RobustnessResult,
    SensitivityResult,
    result_from_json,
)


@dataclass(frozen=True)
class VerificationCheck(Record):
    path: str
    status: Literal["verified", "failed", "unsupported", "invalid"]
    message: str


@dataclass(frozen=True)
class CertificateVerification(Record):
    """Verification relative to saved declarations, not authentication or safety."""

    artifact_ref: ArtifactRef | None
    checks: tuple[VerificationCheck, ...]
    schema_version: ClassVar[str] = "certificate_verification/v1"

    def _validate(self):
        if not self.checks:
            raise ValueError("verification requires at least one check")

    @property
    def status(self):
        for status in ("invalid", "failed", "unsupported"):
            if any(c.status == status for c in self.checks):
                return status
        return "verified"

    @property
    def verified(self):
        return self.status == "verified"

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            **super().to_dict(),
            "status": self.status,
            "verified": self.verified,
        }

    def to_json(self, *, indent=2):
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True, allow_nan=False
        )


class _Verifier:
    def __init__(self, max_checks):
        self.checks, self.max_checks = [], max_checks

    def check(self, path, operation):
        from ._certificate_math import UnsupportedCertificate

        if len(self.checks) >= self.max_checks:
            raise UnsupportedCertificate("Verification check limit exceeded.")
        try:
            value = operation()
            check = VerificationCheck(
                path,
                "verified",
                "Recomputed from the saved declarations with exact rational arithmetic.",
            )
        except UnsupportedCertificate as exc:
            value, check = None, VerificationCheck(path, "unsupported", str(exc))
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            StopIteration,
            OverflowError,
            ZeroDivisionError,
        ) as exc:
            value, check = (
                None,
                VerificationCheck(path, "failed", f"Certificate check failed: {exc}"),
            )
        self.checks.append(check)
        return value

    def result(self, result, path="result", depth=0):
        from ._certificate_math import Context, UnsupportedCertificate
        from ._certificate_results import coverage, evaluation, operating_objective
        from ._certificate_relief import conflict, relief

        if depth > 16:
            raise UnsupportedCertificate("Nested certificate depth limit exceeded.")
        ctx = self.check(path + "/contract", lambda: Context(result.contract))
        if ctx is None:
            return
        if isinstance(result, EvaluationResult):
            self.check(path + "/feasibility", lambda: evaluation(ctx, result))
            feasibility_status = self.checks[-1].status

            def objective_check():
                self.dependencies((feasibility_status,))
                operating_objective(ctx, result)

            self.check(path + "/operating_objective", objective_check)
            known = {"membership", "solver", "conflict", "relief"}
            if result.payload.feasibility == "feasible" or ctx.controller:
                known.add("response")
            if result.payload.feasibility == "infeasible":
                known.add("recourse")
            if ctx.controller:
                known.add("controller")
            if result.payload.objective is not None:
                known.add("operating_objective")
        elif (
            isinstance(result, RobustnessResult)
            and result.request.get("query") == "audit"
        ):
            start = len(self.checks)
            for child in result.supporting_evaluations:
                self.result(child, path + "/evaluations/" + child.result_id, depth + 1)
            dependencies = tuple(c.status for c in self.checks[start:])

            def coverage_check():
                self.dependencies(dependencies)
                coverage(ctx, result)

            self.check(path + "/coverage_and_verdict", coverage_check)
            known = {"coverage", "relief"}
            if result.payload.witness is not None:
                known |= {"membership", "recourse"}
                if ctx.controller:
                    known.add("controller")
            if ctx.controller:
                known |= {"enclosure"} | {
                    f"bound:{c.constraint_id}"
                    for c in ctx.included
                    if c.kind != "equation"
                }
        else:
            self.checks.append(
                VerificationCheck(
                    path,
                    "unsupported",
                    "Only linear evaluations and audits have independent certificate checkers in this version.",
                )
            )
            return
        for item in result.evidence:
            if item.evidence_id == "conflict":
                self.check(path + "/evidence/conflict", lambda: conflict(ctx, result))
            elif item.evidence_id == "relief":
                self.check(
                    path + "/evidence/relief",
                    lambda: relief(
                        ctx,
                        result,
                        lambda nested: self.nested(
                            nested, path + "/evidence/relief/reaudit", depth + 1
                        ),
                    ),
                )
            elif item.evidence_id not in known:
                self.checks.append(
                    VerificationCheck(
                        path + "/evidence/" + item.evidence_id,
                        "unsupported",
                        f"No independent checker for evidence method {item.method!r}.",
                    )
                )
        # Solver logs have no role in the proof. Other unrecognized methods must
        # not acquire trust by borrowing a familiar evidence identity.
        methods = {
            "membership": {"declared_domain_membership"},
            "response": {"exact_physical_residual_check"},
            "recourse": {
                "exact_bounded_linear_contradiction",
                "fixed_controller_violation",
            },
            "controller": {"exact_affine_command_evaluation"},
            "operating_objective": {"exact_separable_lagrangian_bound"},
            "solver": {"scipy_highs_candidates", "cvxpy_prepared_candidates"},
        }
        for item in result.evidence:
            if (
                item.evidence_id in methods
                and item.method not in methods[item.evidence_id]
            ):
                self.checks.append(
                    VerificationCheck(
                        path + "/evidence/" + item.evidence_id,
                        "unsupported",
                        f"Unrecognized evidence method {item.method!r}.",
                    )
                )

    @staticmethod
    def dependencies(statuses):
        from ._certificate_math import UnsupportedCertificate

        statuses = tuple(statuses)
        if any(s in ("failed", "invalid") for s in statuses):
            raise ValueError("A required supporting certificate failed verification.")
        if any(s != "verified" for s in statuses):
            raise UnsupportedCertificate(
                "A required supporting certificate is unsupported."
            )

    def nested(self, result, path, depth):
        start = len(self.checks)
        self.result(result, path, depth)
        self.dependencies(c.status for c in self.checks[start:])


def verify_result(
    document, *, expected_contract_id=None, max_bytes=32_000_000, max_checks=10000
):
    """Check a standard/compact result JSON, mapping, or typed result offline.

    Identity checks are followed by independent exact arithmetic, reconstruction
    of physical rows and objective terms, and coverage checks. No model query,
    production certificate checker or optimizer is executed. Unsupported proof
    families never yield ``verified``. An optional expected contract identity
    pins the declaration against which the exported claims must be checked.
    """
    from ._certificate_math import UnsupportedCertificate
    from collections.abc import Mapping

    verifier = _Verifier(max_checks)
    result = None
    try:
        if (
            type(max_bytes) is not int
            or max_bytes < 1
            or type(max_checks) is not int
            or max_checks < 1
        ):
            raise ValueError("verification limits must be positive integers")
        if expected_contract_id is not None and not isinstance(
            expected_contract_id, str
        ):
            raise ValueError("expected_contract_id must be a string")
        if isinstance(
            document, (EvaluationResult, RobustnessResult, SensitivityResult)
        ):
            document = document.to_json()
        elif isinstance(document, Mapping):
            document = json.dumps(document, allow_nan=False)
        elif isinstance(document, bytes):
            document = document.decode("utf-8")
        if not isinstance(document, str) or len(document.encode("utf-8")) > max_bytes:
            raise ValueError("expected result JSON within max_bytes")
        result = result_from_json(document)
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        return CertificateVerification(
            None, (VerificationCheck("input", "invalid", str(exc)),)
        )
    verifier.checks.append(
        VerificationCheck(
            "input",
            "verified",
            "Result structure, content identities and artifact references are consistent.",
        )
    )
    if (
        expected_contract_id is not None
        and result.contract_ref.artifact_id != expected_contract_id
    ):
        verifier.checks.append(
            VerificationCheck(
                "input/expected_contract",
                "failed",
                "The result is for a different contract than the pinned identity.",
            )
        )
    else:
        try:
            verifier.result(result)
        except (UnsupportedCertificate, RecursionError) as exc:
            verifier.checks.append(VerificationCheck("limits", "unsupported", str(exc)))
    return CertificateVerification(result.ref, tuple(verifier.checks))
