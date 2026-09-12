"""Threshold searches with affine outer bounds and actual controller replay."""

from dataclasses import replace

from ._affine_distance import compile_problem
from ._affine_distance_results import threshold_result as search_threshold
from ._controller_envelope import ControllerEnvelope
from ._controller_results import _failed_constraints, _failure, evaluate_result
from .claim import rejected_result
from .contract_types import Evidence


def threshold_result(
    claim, *, boundary=False, constraints=None, backend="scipy", **options
):
    operation = "boundary" if boundary else "breaking"
    request = {"query": operation, "constraints": constraints, **options}
    known = {
        c.constraint_id
        for c in claim.contract.constraints
        if c.kind == "operating_limit" or c.constraint_id in claim.requirements
    }
    if constraints is not None and (
        not isinstance(constraints, (tuple, list))
        or not constraints
        or any(not isinstance(n, str) or n not in known for n in constraints)
        or len(set(constraints)) != len(constraints)
    ):
        return rejected_result(
            claim.contract,
            operation,
            request,
            "constraints must select distinct included service or equipment constraints",
            code="invalid_distance_search",
            execution="invalid",
        )
    try:
        from ._cvxpy_backend import BackendUnavailable, validate_backend

        validate_backend(backend)
    except (BackendUnavailable, ValueError) as exc:
        return rejected_result(
            claim.contract,
            operation,
            {**request, "backend": backend},
            str(exc),
            code="distance_backend_unavailable",
            execution="unsupported"
            if isinstance(exc, BackendUnavailable)
            else "invalid",
        )
    try:
        envelope = ControllerEnvelope(claim)
        rows = tuple(
            r
            for r in envelope.residuals()
            if constraints is None or r.name in constraints
        )
    except (ValueError, TypeError, OverflowError, RuntimeError) as exc:
        return rejected_result(
            claim.contract,
            operation,
            request,
            str(exc),
            code="controller_enclosure_unresolved",
            execution="unresolved",
        )
    if backend == "cvxpy":
        from ._convex_distance import ConvexThresholdProblem

        compiler = ConvexThresholdProblem
    else:
        compiler = compile_problem
    result = search_threshold(
        claim,
        boundary=boundary,
        requirements=rows,
        compile_branch=lambda c, r, t: compiler(c, r, t, residual_form=r),
        evaluate_response=evaluate_result,
        failure=_failure,
        failure_constraints=_failed_constraints,
        unit=lambda r: r.unit,
        residual_id=claim.adapter.model.residual_id,
        backend=backend,
        **options,
    )
    return replace(
        result,
        evidence=result.evidence
        + (
            Evidence(
                "controller_enclosure",
                "controller_response",
                "exact_substitution_and_command_rounding_bounds",
                "verified",
                details={
                    "controller_ref": claim.adapter.controller.ref.to_dict(),
                    "support_bounds": envelope.proofs,
                    "command_rounding_errors_exact": {
                        n: str(v[2]) for n, v in envelope.commands.items()
                    },
                    "scope": "affine outer targets bound distance from below; upper bounds require exact physical replay",
                },
            ),
        ),
    )
