"""Opt-in conflict reduction and one-limit physical relief, with checked evidence."""

from dataclasses import asdict, replace
from fractions import Fraction
from math import inf, nextafter

from . import linear
from ._linear_recourse import LinearSystem, solve_system
from ._numeric import round_down, round_up
from .contract_types import Evidence
from .linear_process import LinearControl
from .primitives import finite


def validate_options(model, diagnose, relief):
    if not isinstance(diagnose, bool):
        raise ValueError("diagnose must be a boolean")
    if relief is None:
        return
    from collections.abc import Mapping

    if (
        not isinstance(relief, Mapping)
        or set(relief) - {"constraint", "maximum", "tolerance"}
        or not {"constraint", "maximum"} <= set(relief)
    ):
        raise ValueError("relief needs constraint, maximum, and optionally tolerance")
    if relief["constraint"] not in {r.name for r in model.operating_limits}:
        raise ValueError(
            "relief must name an operating limit; service requirements are preserved"
        )
    if (
        finite(relief["maximum"], "maximum relief") <= 0
        or finite(relief.get("tolerance", 1e-8), "relief tolerance") < 0
    ):
        raise ValueError("maximum relief must be positive and tolerance nonnegative")


def conflict_evidence(system, solution, tolerance, *, backend="scipy"):
    current, proof = system, solution.certificate
    removals, unresolved = {}, []
    for name in system.row_refs:
        indices = [i for i, ref in enumerate(current.row_refs) if ref != name]
        trial = replace(
            current,
            rows=[current.rows[i] for i in indices],
            upper=[current.upper[i] for i in indices],
            row_refs=[current.row_refs[i] for i in indices],
        )
        checked = solve_system(trial, tolerance, backend=backend)
        if checked.feasibility == "infeasible":
            current, proof = trial, checked.certificate
        elif checked.feasibility == "feasible":
            removals[name] = {
                "controls": checked.controls,
                "satisfied_row_refs": trial.row_refs,
                "arithmetic": "physical float controls checked against exact rows and control bounds",
            }
        else:
            unresolved.append(name)
    irreducible = not unresolved
    used_bounds = []
    for control, a in zip(current.controls, proof["weighted_coefficients_exact"]):
        if Fraction(a):
            side = "lower" if Fraction(a) > 0 else "upper"
            used_bounds.append(f"control_{side}:{control.name}")
    return Evidence(
        "conflict",
        "recourse_conflict",
        "checked_row_deletion_with_fixed_control_bounds",
        "verified" if irreducible else "unresolved",
        details={
            "constraint_refs": current.row_refs,
            "irreducible": irreducible,
            "minimum_cardinality": False,
            "scope": "irreducible row conflict relative to the declared control box and fixed commands; bounds are retained background assumptions",
            "background_control_bounds": [c.to_dict() for c in system.controls],
            "fixed_controls": system.fixed,
            "certificate_bound_refs": used_bounds,
            "infeasibility_certificate": proof,
            "removal_feasibility": {
                n: removals[n] for n in current.row_refs if n in removals
            },
            "unresolved_removals": unresolved,
        },
    )


def relief_evidence(claim, point, system, options, *, backend="scipy"):
    from ._linear_process_results import _physical_response

    model = claim.adapter
    limit = next(r for r in model.operating_limits if r.name == options["constraint"])
    maximum = finite(options["maximum"], "maximum relief")
    tolerance = finite(options.get("tolerance", 1e-8), "relief tolerance")
    unit = model.output(limit.output).unit
    name = "_operange_relief"
    while name in {c.name for c in model.controls}:
        name += "_"
    control = LinearControl(name, unit, 0, maximum)
    augmented = LinearSystem(
        system.controls + (control,),
        dict(system.fixed),
        [
            row
            + (
                (-Fraction(maximum) / Fraction(limit.residual_scale))
                if ref == limit.name
                else Fraction(0),
            )
            for row, ref in zip(system.rows, system.row_refs)
        ],
        list(system.upper),
        list(system.row_refs),
    )
    details = {
        "constraint": limit.name,
        "unit": unit,
        "original_limit": limit.limit,
        "maximum": maximum,
        "tolerance": tolerance,
        "scope": "minimum one-limit relaxation at this realization under the same operating permissions; full-domain restoration requires re-audit",
        "rows_exact": [[str(a) for a in row] for row in augmented.rows],
        "upper_exact": list(map(str, augmented.upper)),
        "row_constraint_refs": augmented.row_refs,
        "control_coordinates": [
            {"name": c.name, "origin": c.lower, "upper": c.upper}
            for c in augmented.controls
        ],
        "fixed_controls": system.fixed,
    }
    lower, candidate, attempts = Fraction(0), None, []
    certificate = {"method": "nonnegative_relief", "lower_exact": "0"}

    def consider(values):
        nonlocal candidate
        if values is None:
            return
        commands = {n: v for n, v in values.items() if n != name}
        exact_output = model.output(limit.output)._exact_value({**point, **commands})
        needed = max(
            Fraction(0),
            limit.sign * (exact_output - Fraction(limit.limit))
            - Fraction(limit.tolerance),
        )
        changed_exact = Fraction(limit.limit) + limit.sign * needed
        changed_limit = (
            round_up(changed_exact) if limit.sign > 0 else round_down(changed_exact)
        )
        amount = limit.sign * (Fraction(changed_limit) - Fraction(limit.limit))
        if not 0 <= amount <= Fraction(maximum):
            return
        changed = replace(
            model,
            operating_limits=tuple(
                replace(r, limit=changed_limit) if r.name == limit.name else r
                for r in model.operating_limits
            ),
        )
        _, residuals = _physical_response(changed, claim.requirements, point, commands)
        if any(
            residuals[c.constraint_id] > c.tolerance
            for c in changed.base_contract.constraints
            if c.kind != "requirement" or c.constraint_id in claim.requirements
        ):
            return
        if candidate is None or amount < candidate[0]:
            candidate = (
                amount,
                {
                    "changed_limit": changed_limit,
                    "controls": commands,
                    "residuals": residuals,
                    "relief_exact": str(amount),
                },
            )

    try:
        rows = [
            [finite(float(a), "LP coefficient") for a in row] for row in augmented.rows
        ]
        upper = [finite(float(b), "LP rhs") for b in augmented.upper]
        objective = [0.0] * len(system.controls) + [maximum]
        if backend == "cvxpy":
            from ._cvxpy_backend import solve_bounded

            proposal = solve_bounded(augmented, objective, model.solver_tolerance)
            primal, multipliers = proposal.point, proposal.multipliers
            attempts.extend(proposal.attempts)
        else:
            primal, report = linear.solve_lp(
                objective,
                [(0.0, 1.0)] * len(objective),
                inequalities=rows,
                upper=upper,
                tolerance=model.solver_tolerance,
            )
            attempts.append({"purpose": "minimum_relief_candidate", **asdict(report)})
        if primal is not None:
            # A numerical optimum can lie just outside a row. Use its commands
            # at the allowed maximum, then recompute the required physical limit.
            coordinates = list(primal[:-1])
            proposals = [coordinates]
            for i, value in enumerate(coordinates):
                for direction in (-inf, inf):
                    nearby = list(coordinates)
                    nearby[i] = nextafter(value, direction)
                    proposals.append(nearby)
            for proposal in proposals:
                consider(augmented.candidate(proposal + [1.0]))
        n, m = len(objective), len(rows)
        if backend != "cvxpy":
            dual, report = linear.solve_lp(
                upper + [-1.0] * n,
                [(0.0, None)] * m + [(None, 0.0)] * n,
                inequalities=[
                    [-row[j] for row in rows] + [float(k == j) for k in range(n)]
                    for j in range(n)
                ],
                upper=objective,
                tolerance=model.solver_tolerance,
            )
            attempts.append(
                {"purpose": "relief_lower_bound_candidate", **asdict(report)}
            )
            multipliers = dual[:m] if dual is not None else None
        if multipliers is not None:
            bound, proof = augmented.objective_bound(objective, multipliers)
            if bound > lower:
                lower, certificate = bound, proof
        if candidate is None:
            feasible = solve_system(augmented, model.solver_tolerance, backend=backend)
            if feasible.feasibility == "infeasible":
                details.update(
                    {
                        "resolution": "unreachable",
                        "infeasibility_certificate": feasible.certificate,
                        "attempts": attempts + feasible.attempts,
                    }
                )
                return Evidence(
                    "relief",
                    "capacity_relief",
                    "exact_bounded_one_limit_relief",
                    "verified",
                    details=details,
                )
            consider(feasible.controls)
    except (ValueError, OverflowError) as exc:
        details["diagnostic"] = str(exc)
    upper_bound = round_up(candidate[0]) if candidate is not None else None
    lower_bound = round_down(lower)
    inconsistent = upper_bound is not None and lower_bound > upper_bound
    if inconsistent:
        # Conflicting evidence must never establish optimality.
        details["diagnostic"] = "lower certificate contradicts the physical candidate"
        lower_bound, certificate = 0.0, {"method": "nonnegative_relief"}
    minimum = (
        not inconsistent
        and upper_bound is not None
        and Fraction(upper_bound) - Fraction(lower_bound) <= Fraction(tolerance)
    )
    details.update(
        {
            "resolution": "minimum_verified"
            if minimum
            else "bounded"
            if candidate
            else "unresolved",
            "lower": lower_bound,
            "upper": upper_bound,
            "lower_certificate": certificate,
            "candidate": candidate[1] if candidate else None,
            "attempts": attempts,
        }
    )
    return Evidence(
        "relief",
        "capacity_relief",
        "exact_bounded_one_limit_relief",
        "verified" if minimum else "unresolved",
        details=details,
    )
