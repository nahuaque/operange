"""Bounded range search; only checked audits can update engineering bounds."""

from dataclasses import replace
from fractions import Fraction as F

from .claim import Claim
from ._cvxpy_backend import BackendUnavailable
from .contract_types import Diagnostic, snapshot
from .controllers import _ControllerAdapter
from .linear_process import LinearProcessAdapter
from .operating_limits import OperatingLimitResult, OperatingLimitTrial
from .primitives import BoxSet, Parameter, finite


def range_claim(claim, parameter, endpoint):
    domain = BoxSet(
        tuple(
            replace(p, upper=endpoint, nominal=min(p.nominal, endpoint))
            if p.name == parameter
            else p
            for p in claim.domain.parameters
        )
    )
    return replace(claim, domain=domain)


def search(claim, parameter, *, tolerance, max_iterations, max_vertices, backend):
    if not isinstance(claim, Claim):
        raise ValueError("operating-limit search requires a Claim")
    request = snapshot(
        {
            "query": "operating_limit",
            "parameter": parameter,
            "tolerance": tolerance,
            "max_iterations": max_iterations,
            "max_vertices": max_vertices,
            "backend": backend,
            "range_semantics": "all values from the original lower endpoint through the candidate endpoint",
        }
    )
    trials = []

    def result(execution, message=None):
        return OperatingLimitResult(
            claim.contract,
            request,
            execution,
            tuple(trials),
            ()
            if message is None
            else (
                Diagnostic("operating_limit_" + execution, "operating_limit", message),
            ),
        )

    try:
        if not isinstance(parameter, str) or not parameter.strip():
            raise ValueError(
                "parameter must name one scalar production or load coordinate"
            )
        tolerance = finite(tolerance, "operating-limit tolerance")
        if tolerance <= 0:
            raise ValueError(
                "tolerance must be positive in the parameter's physical unit"
            )
        if type(max_iterations) is not int or max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        if type(max_vertices) is not int or max_vertices < 1:
            raise ValueError("max_vertices must be a positive integer")
        if backend not in ("scipy", "cvxpy"):
            raise ValueError("backend must be scipy or cvxpy")
        if type(claim.domain) is not BoxSet:
            raise NotImplementedError(
                "Operating-limit queries currently require a BoxSet."
            )
        p = next((p for p in claim.domain.parameters if p.name == parameter), None)
        if type(p) is not Parameter:
            raise ValueError("parameter must name a top-level scalar box coordinate")
        if type(claim.adapter) not in (LinearProcessAdapter, _ControllerAdapter):
            raise NotImplementedError(
                "Operating-limit queries require a declarative linear process or saved affine controller."
            )
        model = (
            claim.adapter.model
            if type(claim.adapter) is _ControllerAdapter
            else claim.adapter
        )
        if (
            model.measurements is not None
            and parameter in model.measurements.error_inputs
        ):
            raise ValueError(
                "an operating coordinate cannot be a latent sensor-error input"
            )
        if not claim.capabilities.audit.supported:
            raise NotImplementedError(claim.capabilities.audit.scope)
        if type(claim.adapter) is LinearProcessAdapter:
            from ._linear_vertex_audit import vertex_count

            if vertex_count(claim.domain) > max_vertices:
                raise NotImplementedError(
                    "Complete adjustable coverage exceeds max_vertices; no sampling fallback."
                )
            from ._cvxpy_backend import validate_backend

            validate_backend(backend)
    except (NotImplementedError, BackendUnavailable) as exc:
        return result("unsupported", str(exc))
    except (ValueError, TypeError, OverflowError) as exc:
        return result("invalid", str(exc))

    def probe(endpoint):
        candidate = range_claim(claim, parameter, endpoint)
        options = (
            {}
            if type(claim.adapter) is _ControllerAdapter
            else {"backend": backend, "max_vertices": max_vertices}
        )
        audit = candidate.audit_result(**options)
        trial = OperatingLimitTrial(endpoint, audit)
        trials.append(trial)
        return trial.outcome

    try:
        low, high = p.lower, p.upper
        outcome = probe(low)
        if outcome == "fail":
            return result("completed")
        if outcome != "pass":
            return result(
                "unresolved",
                "The operating floor has no conclusive audit; no achievable range was inferred.",
            )
        if low == high:
            return result("completed")
        outcome = probe(high)
        if outcome == "pass":
            return result("completed")
        if outcome != "fail":
            return result(
                "unresolved",
                "The search ceiling is unresolved; it is not an excluded operating point.",
            )
        for _ in range(max_iterations):
            if F(high) - F(low) <= F(tolerance):
                return result("completed")
            midpoint = float((F(low) + F(high)) / 2)
            if not low < midpoint < high:
                return result(
                    "unresolved",
                    "Adjacent representable endpoints prevent further refinement at this tolerance.",
                )
            outcome = probe(midpoint)
            if outcome == "pass":
                low = midpoint
            elif outcome == "fail":
                high = midpoint
            else:
                return result(
                    "unresolved",
                    "An intermediate audit was inconclusive; previously established bounds are retained.",
                )
        if F(high) - F(low) <= F(tolerance):
            return result("completed")
        return result(
            "unresolved",
            "The iteration budget ended before the checked bounds met the requested tolerance.",
        )
    except (ValueError, TypeError, OverflowError, RuntimeError) as exc:
        return result("unresolved", str(exc))
