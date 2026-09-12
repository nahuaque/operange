"""One equipment-relief vector shared by every scenario or hull generator."""

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction as F

from ._dispatch_objective import solve_dispatch
from ._joint_relief import candidate_changes, compile_relief, declarations
from ._linear_recourse import LinearSystem, checked_candidates, compile_system
from ._linear_vertex_audit import vertex_count
from ._numeric import round_down, round_up
from .claim import rejected_result
from .contract_types import Diagnostic, Evidence
from .domains import FiniteSet
from .linear_process import LinearProcessAdapter


def _blocks(claim, baseline, count):
    """Keep all labels; identical physical inputs may share a dispatch block."""
    evaluations = {e.result_id: e for e in baseline.supporting_evaluations}
    blocks = {}
    scenarios = baseline.payload.coverage.evaluated_support["scenarios"]
    if len(scenarios) != count:
        raise ValueError(
            "Shared relief cannot omit any declared scenario or generator."
        )
    for scenario in scenarios:
        evaluation = evaluations[scenario["evaluation_ref"]["artifact_id"]]
        if evaluation.payload.membership.status != "inside":
            raise ValueError("Every relief scenario needs verified domain membership.")
        point = claim.domain.space.validate(evaluation.request["realization"])
        key = tuple(point[n] for n in claim.domain.space.names)
        if key not in blocks:
            blocks[key] = {
                "realization": point,
                "scenario_names": [],
                "system": compile_system(claim, point),
                "evaluation": evaluation,
            }
        blocks[key]["scenario_names"].append(scenario["name"])
    if not blocks:
        raise ValueError("Shared relief requires complete, nonempty scenario coverage.")
    return list(blocks.values())


def _combine(blocks):
    controls, rows, upper, refs, row_scenarios = [], [], [], [], []
    width = sum(len(b["system"].controls) for b in blocks)
    for i, block in enumerate(blocks):
        system = block["system"]
        block["offset"] = offset = len(controls)
        renamed = [
            replace(c, name=f"_operange_dispatch_{i}_{j}")
            for j, c in enumerate(system.controls)
        ]
        block["control_mapping"] = {
            c.name: new.name for c, new in zip(system.controls, renamed)
        }
        controls.extend(renamed)
        for row, bound, ref in zip(system.rows, system.upper, system.row_refs):
            rows.append(
                (F(0),) * offset + tuple(row) + (F(0),) * (width - offset - len(row))
            )
            upper.append(bound)
            refs.append(ref)
            row_scenarios.append(block["scenario_names"])
    return LinearSystem(tuple(controls), {}, rows, upper, refs), row_scenarios


def _at_maximum(model, changes, blocks, augmented, primal):
    """Repair each independent dispatch candidate before checking the joint one."""
    limits = {r.name: r for r in model.operating_limits}
    maximum = {c["constraint"]: F(c["maximum"]) for c in changes}
    proposed = {}
    for block in blocks:
        system = block["system"]
        start = block["offset"]
        coordinates = list(primal[start : start + len(system.controls)])
        relaxed = replace(
            system,
            upper=[
                b + maximum[ref] / F(limits[ref].residual_scale)
                if ref in maximum
                else b
                for b, ref in zip(system.upper, system.row_refs)
            ],
        )
        values = next(iter(checked_candidates(relaxed, coordinates)), None)
        if values is None:
            # A tiny near-bound flow can make an exact balance impossible to
            # express with the other physical floats. Try the actual bound,
            # then reuse row repair and exact checks without relaxing any row.
            snapped = [
                0.0
                if abs(z) <= model.solver_tolerance
                else 1.0
                if abs(z - 1) <= model.solver_tolerance
                else z
                for z in coordinates
            ]
            values = next(iter(checked_candidates(relaxed, snapped)), None)
        if values is None:
            return None
        # Keep the checked physical floats: renormalizing and denormalizing
        # them again can break several independent exact balance equations.
        proposed.update(
            {new: values[old] for old, new in block["control_mapping"].items()}
        )
    proposed.update({c.name: c.upper for c in augmented.controls[-len(changes) :]})
    return proposed


def _candidate(claim, blocks, dispatches, changes, objective, backend, max_vertices):
    from ._linear_process_results import audit_result, evaluate_result

    changed, records, value = candidate_changes(
        claim.adapter,
        changes,
        [(b["realization"], commands) for b, commands in zip(blocks, dispatches)],
        objective,
    )
    by_point = {
        tuple(b["realization"][n] for n in claim.domain.space.names): commands
        for b, commands in zip(blocks, dispatches)
    }

    def replay(changed_claim, realization, **kwargs):
        return evaluate_result(
            changed_claim,
            realization,
            _controls=by_point[tuple(realization[n] for n in claim.domain.space.names)],
            **kwargs,
        )

    # Recompile the changed physical contract and check the proposed commands;
    # do not depend on a second solver finding the same boundary dispatch.
    reaudit = audit_result(
        replace(claim, adapter=changed),
        backend=backend,
        max_vertices=max_vertices,
        _evaluate=replay,
    )
    if reaudit.payload.verdict != "pass":
        raise ValueError(
            "The proposed shared changes failed independent full-domain replay."
        )
    return {
        "changes": records,
        "objective_exact": str(value),
        "dispatches": [
            {
                "scenario_names": b["scenario_names"],
                "realization": b["realization"],
                "controls": commands,
            }
            for b, commands in zip(blocks, dispatches)
        ],
        "reaudit": reaudit.to_dict(compact=True),
        "reaudit_ref": reaudit.ref.to_dict(),
    }


def _relief(claim, baseline, options, changes, *, backend, max_vertices, count):
    details = {
        "declaration": {
            "changes": changes,
            "objective": options["objective"],
            "tolerance": options.get("tolerance", 1e-8),
            "max_scenarios": options.get("max_scenarios", 256),
        },
        "unit": "1",
        "scenario_count": count,
        "scope": "one equipment-relief vector shared by the entire original domain; fixed commands, control bounds, service requirements and observation permissions preserved",
        "formula": "sum(weight * (physical_relief / scale)**2)"
        if options["objective"] == "quadratic"
        else "sum(weight * physical_relief / scale)",
        "arithmetic": "exact_rationals_of_declared_floats; bounds rounded outward",
        "coverage": "complete_finite"
        if type(claim.domain) is FiniteSet
        else "analytical_domain",
        "coverage_proof": "all declared scenarios, or all generators with a common relief vector and convex combinations of their feasible dispatches",
        "dispatch_objective_scope": "equipment relief is optimized; the model's dispatch objective is retained but not used to choose equipment",
    }
    candidate, lower, certificate, attempts = (
        None,
        F(0),
        {"method": "nonnegative_relief_objective"},
        [],
    )
    try:
        blocks = _blocks(claim, baseline, count)
        system, row_scenarios = _combine(blocks)
        augmented, polynomial = compile_relief(
            claim.adapter, system, changes, options["objective"]
        )
        details.update(
            rows_exact=[[str(a) for a in row] for row in augmented.rows],
            upper_exact=list(map(str, augmented.upper)),
            row_constraint_refs=augmented.row_refs,
            row_scenario_names=row_scenarios,
            control_coordinates=[
                {"name": c.name, "origin": c.lower, "upper": c.upper}
                for c in augmented.controls
            ],
            dispatch_blocks=[
                {
                    "scenario_names": b["scenario_names"],
                    "realization": b["realization"],
                    "control_mapping": b["control_mapping"],
                    "fixed_controls": b["system"].fixed,
                }
                for b in blocks
            ],
            objective_linear_exact=list(map(str, polynomial.linear)),
            objective_diagonal_exact=list(map(str, polynomial.diagonal)),
        )
        if baseline.payload.verdict == "pass":
            dispatches = [
                {
                    v.quantity_ref: v.value
                    for v in b["evaluation"].payload.values
                    if v.quantity_ref in {c.name for c in claim.adapter.controls}
                }
                for b in blocks
            ]
        else:
            solution, lower, certificate = solve_dispatch(
                augmented,
                polynomial,
                claim.adapter.solver_tolerance,
                backend=backend,
                candidate_values=lambda primal: _at_maximum(
                    claim.adapter, changes, blocks, augmented, primal
                ),
            )
            attempts = solution.attempts
            if solution.feasibility == "infeasible":
                details.update(
                    resolution="unreachable",
                    infeasibility_certificate=solution.certificate,
                    attempts=attempts,
                )
                return Evidence(
                    "relief",
                    "capacity_relief",
                    "exact_bounded_shared_relief",
                    "verified",
                    details=details,
                )
            if solution.controls is None:
                raise ValueError(solution.message)
            dispatches = [
                {
                    **b["system"].fixed,
                    **{
                        original: solution.controls[renamed]
                        for original, renamed in b["control_mapping"].items()
                    },
                }
                for b in blocks
            ]
        candidate = _candidate(
            claim,
            blocks,
            dispatches,
            changes,
            options["objective"],
            backend,
            max_vertices,
        )
    except (ValueError, OverflowError) as exc:
        details["diagnostic"] = str(exc)
    try:
        upper = (
            round_up(F(candidate["objective_exact"])) if candidate is not None else None
        )
        low = round_down(lower)
    except (ValueError, OverflowError) as exc:
        details["diagnostic"] = str(exc)
        upper, candidate, low = None, None, 0.0
        certificate = {"method": "nonnegative_relief_objective"}
    inconsistent = upper is not None and low > upper
    if inconsistent:
        low, certificate = 0.0, {"method": "nonnegative_relief_objective"}
        details["diagnostic"] = (
            "lower certificate contradicts the full-domain candidate"
        )
    minimum = (
        not inconsistent
        and upper is not None
        and F(upper) - F(low) <= F(options.get("tolerance", 1e-8))
    )
    details.update(
        resolution="minimum_verified"
        if minimum
        else "bounded"
        if candidate
        else "unresolved",
        lower=low,
        upper=upper,
        lower_certificate=certificate,
        candidate=candidate,
        attempts=attempts,
    )
    return Evidence(
        "relief",
        "capacity_relief",
        "exact_bounded_shared_relief",
        "verified" if minimum else "unresolved",
        details=details,
    )


def audit_with_relief(claim, options, *, backend, max_vertices):
    from ._linear_process_results import audit_result

    request = {"query": "audit", "relief": options, "backend": backend}
    if type(claim.domain) is not FiniteSet:
        request["max_vertices"] = max_vertices
    if type(claim.adapter) is not LinearProcessAdapter:
        return rejected_result(
            claim.contract,
            "audit",
            request,
            "Shared relief requires the built-in linear process adapter.",
        )
    try:
        if not isinstance(options, Mapping) or "changes" not in options:
            raise ValueError(
                "Shared relief needs changes and an explicit linear or quadratic objective."
            )
        changes = declarations(
            claim.adapter, {k: v for k, v in options.items() if k != "max_scenarios"}
        )
        limit = options.get("max_scenarios", 256)
        if type(limit) is not int or limit < 1:
            raise ValueError("max_scenarios must be a positive integer.")
    except (ValueError, TypeError, KeyError) as exc:
        return rejected_result(
            claim.contract,
            "audit",
            request,
            str(exc),
            code="invalid_relief_options",
            execution="invalid",
        )
    if options["objective"] == "quadratic" and backend != "cvxpy":
        return rejected_result(
            claim.contract,
            "audit",
            request,
            "Quadratic shared relief requires backend='cvxpy'.",
            code="relief_backend_unavailable",
        )
    count = (
        len(claim.domain.scenarios)
        if type(claim.domain) is FiniteSet
        else vertex_count(claim.domain)
    )
    if count > limit:
        return rejected_result(
            claim.contract,
            "audit",
            request,
            f"Shared relief needs all {count} scenarios/generators, exceeding max_scenarios={limit}; none were evaluated.",
            code="relief_scenario_limit_exceeded",
        )
    baseline = audit_result(claim, backend=backend, max_vertices=max_vertices)
    if baseline.execution in ("invalid", "unsupported"):
        return replace(baseline, request=request)
    proof = _relief(
        claim,
        baseline,
        options,
        changes,
        backend=backend,
        max_vertices=max_vertices,
        count=count,
    )
    resolved = proof.details["resolution"] in ("minimum_verified", "unreachable")
    return replace(
        baseline,
        request=request,
        evidence=baseline.evidence + (proof,),
        execution=baseline.execution if resolved else "unresolved",
        diagnostics=baseline.diagnostics
        + (
            ()
            if resolved
            else (
                Diagnostic(
                    "shared_relief_unresolved",
                    "relief",
                    proof.details.get(
                        "diagnostic",
                        "The shared-relief objective gap remains open; inspect the checked bounds and candidate re-audit.",
                    ),
                ),
            )
        ),
    )
