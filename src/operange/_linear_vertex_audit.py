"""Continuous linear recourse coverage from all declared domain generators."""

from dataclasses import replace
from functools import partial
from itertools import product

from ._finite_audit import audit_finite
from .claim import rejected_result
from .convex_hull import ConvexHullSet
from .domains import Scenario
from .primitives import BoxSet


def vertex_count(domain):
    if type(domain) is BoxSet:
        return 2 ** sum(p.lower != p.upper for p in domain.scalar_parameters)
    if type(domain) is ConvexHullSet:
        return len(domain.vertices)
    raise ValueError(
        "Continuous recourse requires a BoxSet or an explicit ConvexHullSet."
    )


def audit_vertices(claim, *, backend, max_vertices, evaluate=None):
    from ._linear_process_results import evaluate_result, _failure, _failed_constraints

    request = {"query": "audit", "backend": backend, "max_vertices": max_vertices}
    if type(max_vertices) is not int or max_vertices < 1:
        return rejected_result(
            claim.contract,
            "audit",
            request,
            "max_vertices must be a positive integer.",
            code="invalid_vertex_limit",
            execution="invalid",
        )
    domain = claim.domain
    if type(domain) is BoxSet:
        parameters = sorted(domain.scalar_parameters, key=lambda p: p.name)
        count = vertex_count(domain)
        kind = "all_box_corners"
    elif type(domain) is ConvexHullSet:
        count = vertex_count(domain)
        kind = "all_declared_hull_generators"
    else:
        return rejected_result(
            claim.contract,
            "audit",
            request,
            "Continuous recourse requires a BoxSet or an explicit ConvexHullSet.",
        )
    if count > max_vertices:
        return rejected_result(
            claim.contract,
            "audit",
            request,
            f"Complete coverage requires {count} generators, exceeding max_vertices={max_vertices}. "
            "No generators were evaluated; increase the explicit limit to enumerate them all.",
            code="vertex_limit_exceeded",
        )
    if type(domain) is BoxSet:
        vertices = (
            Scenario(
                f"corner_{i}",
                dict(zip((p.name for p in parameters), values)),
                "Exhaustive physical box endpoints; constant coordinates appear once",
            )
            for i, values in enumerate(
                product(
                    *(
                        (p.lower,) if p.lower == p.upper else (p.lower, p.upper)
                        for p in parameters
                    )
                )
            )
        )
    else:
        vertices = domain.vertices
    scope = {
        "controls": "fixed as declared or independently adjustable after all inputs are observed",
        "coverage": "continuous convex hull of every enumerated generator",
        "generator_kind": kind,
        "generator_count": count,
        "objective_scope": "feasibility throughout the domain; any objective bounds apply only to individual generator evaluations",
        "controller_scope": "existence of real-valued convex combinations of checked controls; no executable or rounded controller asserted",
    }
    result = audit_finite(
        claim,
        partial(evaluate_result if evaluate is None else evaluate, backend=backend),
        _failure,
        generators=vertices,
        coverage_method="analytical_domain",
        method="complete_vertex_linear_recourse",
        scope=scope,
        coverage_details={
            **scope,
            "domain_ref": domain.ref.to_dict(),
            "arithmetic": "exact_rationals_of_declared_floats",
            "assumptions": "fixed affine input/control coefficients, constant bounds and tolerances, continuous bounded controls, identical fixed commands and selected requirements at all generators",
            "proof": "For weights w_i >= 0 summing to 1, u = sum(w_i * u_i) and c = sum(w_i * c_i). Every affine residual at (u,c) is the same convex combination of checked generator residuals, hence stays within its declared tolerance. Control bounds and fixed commands are preserved.",
            "failure_scope": "one certified infeasible generator refutes the continuous claim; otherwise every generator must have a verified feasible response to pass",
        },
        failure_constraints=_failed_constraints,
        diagnostic="A generator has neither a verified dispatch nor an infeasibility certificate; continuous coverage is unresolved.",
    )
    return replace(result, request=request)
