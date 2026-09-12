"""Joint operating-limit relief through the existing bounded objective solver."""

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction as F

from ._dispatch_objective import Polynomial, solve_dispatch
from ._linear_recourse import LinearSystem
from ._numeric import round_down, round_up
from .contract_types import Evidence
from .linear_process import LinearControl
from .primitives import finite


def declarations(model, options):
    if set(options) - {"changes", "objective", "tolerance"} or options.get(
        "objective"
    ) not in ("linear", "quadratic"):
        raise ValueError(
            "joint relief requires changes and an explicit linear or quadratic objective"
        )
    changes = options["changes"]
    if not isinstance(changes, (tuple, list)) or not changes:
        raise ValueError("joint relief changes must be a nonempty sequence")
    if finite(options.get("tolerance", 1e-8), "relief tolerance") < 0:
        raise ValueError("relief tolerance must be nonnegative")
    known = {r.name: r for r in model.operating_limits}
    result, seen = [], set()
    for change in changes:
        required = {"constraint", "maximum", "scale", "unit"}
        if (
            not isinstance(change, Mapping)
            or not required <= set(change)
            or set(change) - required - {"weight"}
        ):
            raise ValueError(
                "each relief change requires constraint, maximum, scale, unit and optionally weight"
            )
        name = change["constraint"]
        if name not in known or name in seen:
            raise ValueError(
                "relief changes must name distinct operating limits; service requirements are preserved"
            )
        seen.add(name)
        limit = known[name]
        if change["unit"] != model.output(limit.output).unit:
            raise ValueError(
                "relief scale and maximum must use the operating limit's unit"
            )
        numbers = {
            k: finite(change.get(k, 1), f"relief {k}")
            for k in ("maximum", "scale", "weight")
        }
        if any(v <= 0 for v in numbers.values()):
            raise ValueError("relief maximum, scale and weight must be positive")
        result.append({"constraint": name, "unit": change["unit"], **numbers})
    return tuple(result)


def relief_evidence(claim, point, system, options, *, backend):
    from ._linear_process_results import _physical_response

    model = claim.adapter
    changes = declarations(model, options)
    limits = {r.name: r for r in model.operating_limits}
    controls, names = [], {c.name for c in model.controls}
    for i, change in enumerate(changes):
        name = f"_operange_relief_{i}"
        while name in names:
            name += "_"
        names.add(name)
        controls.append(LinearControl(name, change["unit"], 0, change["maximum"]))
    augmented = LinearSystem(
        system.controls + tuple(controls),
        dict(system.fixed),
        [
            row
            + tuple(
                -F(c["maximum"]) / F(limits[ref].residual_scale)
                if ref == c["constraint"]
                else F(0)
                for c in changes
            )
            for row, ref in zip(system.rows, system.row_refs)
        ],
        list(system.upper),
        list(system.row_refs),
    )
    quadratic = options["objective"] == "quadratic"
    costs = tuple(
        F(c["weight"]) * (F(c["maximum"]) / F(c["scale"])) ** (2 if quadratic else 1)
        for c in changes
    )
    zero = (F(0),) * len(augmented.controls)
    polynomial = Polynomial(
        F(0),
        zero if quadratic else (F(0),) * len(system.controls) + costs,
        (F(0),) * len(system.controls) + costs if quadratic else zero,
    )
    details = {
        "declaration": {
            "changes": changes,
            "objective": options["objective"],
            "tolerance": options.get("tolerance", 1e-8),
        },
        "unit": "1",
        "scope": "minimum declared dimensionless joint relief at this realization; original permissions and service requirements preserved; changed equipment requires a separate full-domain re-audit",
        "formula": "sum(weight * (physical_relief / scale)**2)"
        if quadratic
        else "sum(weight * physical_relief / scale)",
        "arithmetic": "exact_rationals_of_declared_floats; bounds rounded outward",
        "rows_exact": [[str(a) for a in row] for row in augmented.rows],
        "upper_exact": list(map(str, augmented.upper)),
        "row_constraint_refs": augmented.row_refs,
        "control_coordinates": [
            {"name": c.name, "origin": c.lower, "upper": c.upper}
            for c in augmented.controls
        ],
        "fixed_controls": system.fixed,
        "objective_linear_exact": list(map(str, polynomial.linear)),
        "objective_diagonal_exact": list(map(str, polynomial.diagonal)),
    }
    candidate, lower, certificate, attempts = (
        None,
        F(0),
        {"method": "nonnegative_relief_objective"},
        [],
    )
    try:
        solution, lower, certificate = solve_dispatch(
            augmented,
            polynomial,
            model.solver_tolerance,
            backend=backend,
            candidate_transform=lambda primal: (
                list(primal[: len(system.controls)]) + [1.0] * len(changes)
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
                "exact_bounded_joint_relief",
                "verified",
                details=details,
            )
        if solution.controls is not None:
            commands = {c.name: solution.controls[c.name] for c in model.controls}
            updates, records, value = {}, [], F(0)
            for change in changes:
                limit = limits[change["constraint"]]
                response = model.output(limit.output)._exact_value(
                    {**point, **commands}
                )
                needed = max(
                    F(0), limit.sign * (response - F(limit.limit)) - F(limit.tolerance)
                )
                adjusted = F(limit.limit) + limit.sign * needed
                adjusted = (
                    round_up(adjusted) if limit.sign > 0 else round_down(adjusted)
                )
                amount = limit.sign * (F(adjusted) - F(limit.limit))
                if not 0 <= amount <= F(change["maximum"]):
                    raise ValueError(
                        "representable changed limit exceeds the permitted relief"
                    )
                updates[limit.name] = adjusted
                value += F(change["weight"]) * (amount / F(change["scale"])) ** (
                    2 if quadratic else 1
                )
                records.append(
                    {
                        **change,
                        "original_limit": limit.limit,
                        "changed_limit": adjusted,
                        "physical_relief": round_up(amount),
                        "relief_exact": str(amount),
                    }
                )
            changed = replace(
                model,
                operating_limits=tuple(
                    replace(r, limit=updates.get(r.name, r.limit))
                    for r in model.operating_limits
                ),
            )
            _, residuals = _physical_response(
                changed, claim.requirements, point, commands
            )
            if any(
                residuals[c.constraint_id] > c.tolerance
                for c in changed.base_contract.constraints
                if c.kind != "requirement" or c.constraint_id in claim.requirements
            ):
                raise ValueError(
                    "joint relief candidate failed original service or equipment constraints"
                )
            candidate = {
                "changes": records,
                "controls": commands,
                "residuals": residuals,
                "objective_exact": str(value),
            }
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
        details["diagnostic"] = "lower certificate contradicts the physical candidate"
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
        "exact_bounded_joint_relief",
        "verified" if minimum else "unresolved",
        details=details,
    )
