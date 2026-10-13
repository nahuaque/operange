"""Request dispatch and preflight; result construction stays with each study."""

from dataclasses import replace
from importlib.util import find_spec
from typing import get_args

from .claim import Claim
from .domains import FiniteSet
from .studies import (
    AuditRequest,
    BoundaryRequest,
    BreakingRequest,
    ControllerSynthesisRequest,
    DispatchSensitivityRequest,
    EvaluationRequest,
    OperatingLimitRequest,
    SensitivityRequest,
    SharedControlsRequest,
    StudyCapability,
    UncertaintyPropagationRequest,
    EvidenceRequirement,
)


_BASIC = (
    EvaluationRequest,
    SensitivityRequest,
    AuditRequest,
    BoundaryRequest,
    BreakingRequest,
)
_CLAIM = _BASIC + (
    ControllerSynthesisRequest,
    OperatingLimitRequest,
    DispatchSensitivityRequest,
)
_ADAPTER = (SharedControlsRequest, UncertaintyPropagationRequest)
_EVIDENCE = frozenset(get_args(EvidenceRequirement))
_QUALIFICATION = (
    "Support is permission to attempt the study, not a guarantee of feasibility, optimality or completed evidence.",
    "Membership, solver availability and outcome-dependent checks remain part of execution.",
)


def basic_arguments(request):
    args = request.arguments()
    point = args.pop("realization", None)
    return point, {k: v for k, v in args.items() if v is not None}


def _report(request, scope, coverage, evidence, dependencies=(), limitations=()):
    return StudyCapability(
        request.operation,
        "supported",
        scope,
        coverage,
        tuple(sorted(evidence)),
        dependencies,
        _QUALIFICATION + limitations,
    )


def _backend(backend):
    if backend not in ("scipy", "cvxpy"):
        raise ValueError("backend must be 'scipy' or 'cvxpy'")
    return ("cvxpy",) if backend == "cvxpy" else ()


def _basic(claim, request):
    from ._study_validation import audit_dependencies, validate_basic

    capability = getattr(claim.capabilities, request.operation)
    if not capability.supported:
        # Coarse domain support can already be false because its backend is
        # absent. Preserve that declaration before request-specific validation.
        return StudyCapability(
            request.operation,
            "unsupported",
            capability.scope,
            dependencies=audit_dependencies(claim)
            if request.operation == "audit"
            else (),
            limitations=_QUALIFICATION,
        )
    point, options = basic_arguments(request)
    declaration = getattr(claim.adapter, "query_options", None)
    if declaration is None:
        if options:
            raise NotImplementedError(
                "This adapter does not declare request-specific query options."
            )
        return _report(
            request,
            capability.scope,
            "adapter_declared",
            (),
            limitations=(
                "The adapter declares operation support but no structured evidence capabilities.",
            ),
        )
    unknown = set(options) - set(declaration(claim, request.operation))
    if unknown:
        raise ValueError(
            "Unsupported options for this adapter query: " + ", ".join(sorted(unknown))
        )
    coverage, evidence, dependencies = validate_basic(
        claim, request.operation, point, options
    )
    return _report(request, capability.scope, coverage, evidence, dependencies)


def _claim_study(claim, request):
    if type(request) in _BASIC:
        return _basic(claim, request)
    if type(request) is ControllerSynthesisRequest:
        from ._controller_synthesis import _validate
        from .primitives import BoxSet
        from .objectives import LinearObjective

        _, objective, _ = _validate(
            claim,
            request.objective,
            request.name,
            request.max_vertices,
            materialize=False,
        )
        continuous = type(claim.domain) is not FiniteSet
        evidence = {
            "checked_feasibility",
            "controller_audit",
            "independent_certificate",
        }
        if continuous:
            evidence.add("continuous_coverage")
        support = type(claim.domain) is BoxSet and type(objective) is LinearObjective
        return _report(
            request,
            "Affine policy proposals followed by a separate rounded-command audit.",
            "continuous_domain" if continuous else "finite_scenarios",
            evidence,
            ("cvxpy",),
            (
                "Whole-box support; max_vertices does not limit this formulation."
                if support
                else "Complete generators only, subject to max_vertices; no sampling fallback.",
                "No global optimum over controllers or policy-class infeasibility proof.",
            ),
        )
    if type(request) is OperatingLimitRequest:
        from ._operating_limit import prepare
        from .linear_process import LinearProcessAdapter
        from .objectives import ControlTrackingObjective

        prepare(claim, **request.arguments(), check_backend=False)
        dependencies = (
            _backend(request.backend)
            if type(claim.adapter) is LinearProcessAdapter
            else ()
        )
        if (
            type(claim.adapter) is LinearProcessAdapter
            and isinstance(claim.adapter.objective, ControlTrackingObjective)
            and request.backend != "cvxpy"
        ):
            raise NotImplementedError("Control tracking requires backend='cvxpy'.")
        return _report(
            request,
            "Checked audits bound the guaranteed prefix of one box coordinate.",
            "continuous_domain",
            ("range_bounds", "continuous_coverage", "independent_certificate"),
            dependencies,
        )
    from ._dispatch_sensitivity import prepare

    dependencies = _backend(request.backend)
    prepare(claim, **request.arguments(), check_backend=False)
    return _report(
        request,
        "Exact local objective-value slopes and one-sided exclusions for a physical parametric LP.",
        "local",
        ("local_sensitivity", "independent_certificate"),
        dependencies,
        (
            "Requires a checked feasible baseline at execution; selected-command derivatives are not supplied.",
        ),
    )


def _adapter_study(adapter, request):
    # Import optional integration code only for an explicitly requested study.
    from .integrations.idaes import IdaesAdapter

    if not isinstance(adapter, IdaesAdapter):
        raise NotImplementedError(
            "This study requires an IdaesAdapter; bind ordinary queries through a Claim."
        )
    if type(request) is SharedControlsRequest:
        from .integrations._idaes_shared_controls import prepare

        prepare(adapter, **request.arguments())
        return _report(
            request,
            "One shared command vector with source residual checks at every declared scenario.",
            "finite_scenarios",
            ("checked_feasibility", "shared_commands"),
            ("idaes", "pyomo"),
            (
                "Numerical candidates only; no nonlinear infeasibility or global-optimality certificate.",
                "Solver and initialization settings come from the adapter binding.",
            ),
        )
    from .integrations._idaes_uncertainty import prepare
    from .operating import OperatingSpec

    if not isinstance(request.operating, OperatingSpec):
        raise ValueError("Local propagation requests need an explicit OperatingSpec.")
    prepare(adapter, **request.arguments(), fixed_controls=None, reoptimize=False)
    return _report(
        request,
        "Local covariance propagated through an explicit fixed or reoptimized response.",
        "local",
        ("local_covariance", "local_sensitivity"),
        ("idaes", "pyomo"),
        (
            "State regularity and derivative agreement are checked during execution.",
            "Local numerical information supplies no nonlinear confidence coverage or global certificate.",
        ),
    )


def check(target, request):
    operation = getattr(request, "operation", "unknown")
    report = None
    try:
        if type(request) not in _CLAIM + _ADAPTER:
            raise ValueError("Supply one of the declared study request types.")
        required = request.required_evidence
        if (
            not isinstance(required, tuple)
            or any(not isinstance(n, str) or n not in _EVIDENCE for n in required)
            or len(set(required)) != len(required)
        ):
            raise ValueError(
                "required_evidence must contain distinct declared evidence names."
            )
        if type(request) in _CLAIM:
            if not isinstance(target, Claim):
                raise ValueError(
                    "This request must be bound to a Claim, including its domain and operating policy."
                )
            report = _claim_study(target, request)
        else:
            report = _adapter_study(target, request)
        missing = set(required) - set(report.evidence)
        if missing:
            raise NotImplementedError(
                "This study cannot supply required evidence: "
                + ", ".join(sorted(missing))
            )
        missing = [name for name in report.dependencies if find_spec(name) is None]
        if missing:
            raise NotImplementedError(
                "Missing optional study dependencies: " + ", ".join(missing)
            )
        return report
    except (NotImplementedError, ImportError) as exc:
        status = "unsupported"
        message = str(exc)
    except (ValueError, TypeError, KeyError, OverflowError, AttributeError) as exc:
        status = "invalid"
        message = str(exc)
    if report is not None:
        return replace(report, status=status, scope=message)
    return StudyCapability(operation, status, message)


def run(target, request):
    if type(request) in _BASIC:
        point, options = basic_arguments(request)
        return target._run(request.operation, point, **options)
    method = {
        ControllerSynthesisRequest: "synthesize_controller",
        OperatingLimitRequest: "operating_limit",
        DispatchSensitivityRequest: "dispatch_sensitivity",
        SharedControlsRequest: "optimize_shared_controls",
        UncertaintyPropagationRequest: "propagate_uncertainty",
    }[type(request)]
    return getattr(target, method)(**request.arguments())
