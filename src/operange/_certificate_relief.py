"""Independent conflict and equipment-relief checks against original declarations."""

from dataclasses import replace
from fractions import Fraction as F
import json

from ._certificate_math import (
    Context,
    System,
    dot,
    exact_equal,
    q,
    require,
    rounded,
    vector_equal,
)
from ._certificate_results import evidence
from .engineering_results import result_from_json
from .contract_types import plain


def conflict(ctx, result):
    data = evidence(
        result, "conflict", "checked_row_deletion_with_fixed_control_bounds"
    )
    system = ctx.system(result.request["realization"])
    refs = list(data["constraint_refs"])
    require(
        refs and len(refs) == len(set(refs)) and set(refs) <= set(system.refs),
        "invalid conflict row identities",
    )
    indices = [i for i, n in enumerate(system.refs) if n in refs]
    subset = replace(
        system,
        rows=[system.rows[i] for i in indices],
        rhs=[system.rhs[i] for i in indices],
        refs=[system.refs[i] for i in indices],
    )
    subset.contradiction(data["infeasibility_certificate"])
    require(
        data["minimum_cardinality"] is False,
        "row deletion does not prove minimum cardinality",
    )
    require(
        dict(data["fixed_controls"]) == system.fixed, "conflict changes fixed commands"
    )
    require(
        list(data["background_control_bounds"])
        == [c.to_dict() for c in ctx.model.controls if c.name not in system.fixed],
        "conflict changes background bounds",
    )
    used = [
        f"control_{'lower' if q(a) > 0 else 'upper'}:{n}"
        for (n, _, _), a in zip(
            system.coordinates,
            data["infeasibility_certificate"]["weighted_coefficients_exact"],
        )
        if q(a)
    ]
    require(
        list(data["certificate_bound_refs"]) == used, "conflict bound references differ"
    )
    removals = data["removal_feasibility"]
    for name, record in removals.items():
        kept = list(record["satisfied_row_refs"])
        require(
            name in refs
            and name not in kept
            and set(refs) - {name} <= set(kept) <= set(system.refs),
            "removal witness omits a remaining conflict row",
        )
        indices = [i for i, n in enumerate(system.refs) if n in kept]
        trial = replace(
            system,
            rows=[system.rows[i] for i in indices],
            rhs=[system.rhs[i] for i in indices],
            refs=[system.refs[i] for i in indices],
        )
        require(trial.refs == kept, "removal witness row identities differ")
        trial.feasible(record["controls"])
    if data["irreducible"]:
        require(
            set(removals) == set(refs) and not data["unresolved_removals"],
            "irreducibility lacks a feasible witness for every row removal",
        )


def declarations(ctx, options, single):
    limits = {r.name: r for r in ctx.model.operating_limits}
    changes = (
        [
            {
                "constraint": options["constraint"],
                "maximum": options["maximum"],
                "scale": 1,
                "weight": 1,
                "unit": ctx.model.output(limits[options["constraint"]].output).unit,
            }
        ]
        if single
        else options["changes"]
    )
    result = []
    for item in changes:
        name = item["constraint"]
        require(
            name in limits and name not in {c["constraint"] for c in result},
            "relief must change distinct declared equipment limits",
        )
        require(
            item["unit"] == ctx.model.output(limits[name].output).unit,
            "relief units differ",
        )
        record = {k: item[k] for k in ("constraint", "unit", "maximum", "scale")}
        record["weight"] = item.get("weight", 1)
        require(
            all(q(record[k]) > 0 for k in ("maximum", "scale", "weight")),
            "invalid relief maximum, scale or weight",
        )
        result.append(record)
    require(
        result and q(options.get("tolerance", 1e-8)) >= 0, "invalid relief tolerance"
    )
    return result


def combine(ctx, data):
    blocks = {}
    for name, point in ctx.generators():
        key = tuple(point[n] for n in ctx.domain.space.names)
        if key not in blocks:
            blocks[key] = {
                "scenario_names": [],
                "realization": point,
                "system": ctx.system(point),
            }
        blocks[key]["scenario_names"].append(name)
    blocks = list(blocks.values())
    width = sum(len(b["system"].coordinates) for b in blocks)
    system = System([], {}, [], [], [])
    row_names, records = [], []
    for i, block in enumerate(blocks):
        local, offset = block["system"], len(system.coordinates)
        mapping = {
            n: f"_operange_dispatch_{i}_{j}"
            for j, (n, _, _) in enumerate(local.coordinates)
        }
        system.coordinates.extend(
            (mapping[n], lo, span) for n, lo, span in local.coordinates
        )
        for row in local.rows:
            system.rows.append(
                [F(0)] * offset + row + [F(0)] * (width - offset - len(row))
            )
            row_names.append(block["scenario_names"])
        system.rhs.extend(local.rhs)
        system.refs.extend(local.refs)
        records.append(
            {
                "scenario_names": block["scenario_names"],
                "realization": block["realization"],
                "control_mapping": mapping,
                "fixed_controls": local.fixed,
            }
        )
    require(
        plain(data["dispatch_blocks"]) == records,
        "shared relief omits or changes a scenario block or its permissions",
    )
    require(
        plain(data["row_scenario_names"]) == row_names,
        "shared relief rows refer to the wrong scenarios",
    )
    return system, blocks


def relief(ctx, result, verify_nested):
    proof = next(e for e in result.evidence if e.evidence_id == "relief")
    require(
        ctx.controller is None,
        "equipment relief requires adjustable or fixed linear permissions",
    )
    single = proof.method == "exact_bounded_one_limit_relief"
    shared = proof.method == "exact_bounded_shared_relief"
    require(
        proof.method
        in (
            "exact_bounded_one_limit_relief",
            "exact_bounded_joint_relief",
            "exact_bounded_shared_relief",
        ),
        "unknown relief certificate",
    )
    data, options = proof.details, result.request["relief"]
    changes = declarations(ctx, options, single)
    quadratic = not single and options["objective"] == "quadratic"
    require(
        single or options["objective"] in ("linear", "quadratic"),
        "invalid relief objective",
    )
    if single:
        require(
            data["constraint"] == changes[0]["constraint"]
            and data["maximum"] == changes[0]["maximum"]
            and data["unit"] == changes[0]["unit"]
            and data["tolerance"] == options.get("tolerance", 1e-8),
            "relief declaration differs",
        )
    else:
        declaration = {
            "changes": changes,
            "objective": options["objective"],
            "tolerance": options.get("tolerance", 1e-8),
        }
        if shared:
            declaration["max_scenarios"] = options.get("max_scenarios", 256)
        require(
            plain(data["declaration"]) == declaration and data["unit"] == "1",
            "relief declaration differs",
        )
    if shared:
        require(result.request["query"] == "audit", "shared relief must bind an audit")
        system, blocks = combine(ctx, data)
        require(
            data["scenario_count"] == len(ctx.generators()),
            "shared relief scenario count differs",
        )
    else:
        system = ctx.system(result.request["realization"])
        blocks = [{"realization": result.request["realization"], "system": system}]
    limits = {r.name: r for r in ctx.model.operating_limits}
    if single:
        require(
            data["original_limit"] == limits[changes[0]["constraint"]].limit,
            "original equipment limit differs",
        )
    n = len(system.coordinates)
    coordinates = list(system.coordinates)
    names = {name for name, _, _ in coordinates} | set(system.fixed)
    for i, change in enumerate(changes):
        name = "_operange_relief" if single else f"_operange_relief_{i}"
        while name in names:
            name += "_"
        names.add(name)
        coordinates.append((name, F(0), q(change["maximum"])))
    augmented = replace(
        system,
        coordinates=coordinates,
        rows=[
            row
            + [
                -q(c["maximum"]) / q(limits[ref].residual_scale)
                if ref == c["constraint"]
                else F(0)
                for c in changes
            ]
            for row, ref in zip(system.rows, system.refs)
        ],
    )
    augmented.match(data, exact_coordinates=False)
    costs = [
        q(c["maximum"])
        if single
        else q(c["weight"])
        * (q(c["maximum"]) / q(c["scale"])) ** (2 if quadratic else 1)
        for c in changes
    ]
    linear = [F(0)] * n + ([F(0)] * len(changes) if quadratic else costs)
    diagonal = [F(0)] * n + (costs if quadratic else [F(0)] * len(changes))
    if not single:
        vector_equal(
            data["objective_linear_exact"], linear, "relief linear objective differs"
        )
        vector_equal(
            data["objective_diagonal_exact"],
            diagonal,
            "relief quadratic objective differs",
        )
    if data["resolution"] == "unreachable":
        augmented.contradiction(data["infeasibility_certificate"])
        return
    certificate = data["lower_certificate"]
    if certificate.get("method") in (
        "nonnegative_relief",
        "nonnegative_relief_objective",
    ):
        lower = F(0)
        if "lower_exact" in certificate:
            exact_equal(
                certificate["lower_exact"], lower, "nonnegative relief bound differs"
            )
    elif single:
        multipliers = [q(v) for v in certificate["multipliers"]]
        require(
            len(multipliers) == len(augmented.rows)
            and all(v >= 0 for v in multipliers),
            "invalid relief bound multipliers",
        )
        slopes = [
            c + dot([row[i] for row in augmented.rows], multipliers)
            for i, c in enumerate(linear)
        ]
        minimum, rhs = (
            sum((min(0, a) for a in slopes), F(0)),
            dot(augmented.rhs, multipliers),
        )
        lower = minimum - rhs
        vector_equal(
            certificate["objective_exact"], linear, "relief bound objective differs"
        )
        vector_equal(
            certificate["weighted_residual_exact"], slopes, "relief bound slopes differ"
        )
        for key, value in (
            ("box_minimum_exact", minimum),
            ("weighted_upper_exact", rhs),
            ("lower_exact", lower),
        ):
            exact_equal(certificate[key], value, f"{key} differs")
    else:
        lower = augmented.lower_bound(F(0), linear, diagonal, certificate)
    require(data["lower"] == rounded(lower, -1), "reported relief lower bound differs")
    candidate = data["candidate"]
    if candidate is None:
        require(
            data["upper"] is None and data["resolution"] == "unresolved",
            "relief upper bound lacks a physical candidate",
        )
        return
    records = (
        [{"changed_limit": candidate["changed_limit"]}]
        if single
        else candidate["changes"]
    )
    require(len(records) == len(changes), "candidate changes omit or add equipment")
    updates, value = {}, F(0)
    for change, record in zip(changes, records):
        row = limits[change["constraint"]]
        amount = (1 if row.relation == "le" else -1) * (
            q(record["changed_limit"]) - q(row.limit)
        )
        require(
            0 <= amount <= q(change["maximum"]),
            "candidate exceeds permitted equipment relief",
        )
        updates[row.name] = record["changed_limit"]
        value += (
            amount
            if single
            else q(change["weight"])
            * (amount / q(change["scale"])) ** (2 if quadratic else 1)
        )
        if not single:
            require(
                all(record[k] == v for k, v in change.items())
                and record["original_limit"] == row.limit,
                "candidate changes a relief declaration",
            )
            exact_equal(
                record["relief_exact"], amount, "physical relief amount differs"
            )
            require(
                record["physical_relief"] == rounded(amount, 1),
                "exported physical relief differs",
            )
    changed_model = replace(
        ctx.model,
        operating_limits=tuple(
            replace(r, limit=updates.get(r.name, r.limit))
            for r in ctx.model.operating_limits
        ),
    )
    changed = Context(replace(ctx.claim, adapter=changed_model).contract)
    if shared:
        dispatches = candidate["dispatches"]
        require(len(dispatches) == len(blocks), "candidate omits dispatch blocks")
        for block, dispatch in zip(blocks, dispatches):
            require(
                dispatch["scenario_names"] == tuple(block["scenario_names"])
                and dict(dispatch["realization"]) == dict(block["realization"]),
                "candidate dispatch refers to different scenarios",
            )
            changed.system(block["realization"]).feasible(dispatch["controls"])
        nested = result_from_json(json.dumps(plain(candidate["reaudit"])))
        require(
            nested.contract == changed.claim.contract
            and candidate["reaudit_ref"] == nested.ref.to_dict()
            and nested.payload.verdict == "pass",
            "changed audit is not for the common equipment plan",
        )
        for child in nested.supporting_evaluations:
            dispatch = next(
                d
                for d in dispatches
                if d["realization"] == child.request["realization"]
            )
            require(
                child.request.get("dispatch_source") == "shared_relief_candidate"
                and child.request["controls"] == dispatch["controls"],
                "changed audit did not replay the proposed commands",
            )
        verify_nested(nested)
    else:
        point = result.request["realization"]
        changed.system(point).feasible(candidate["controls"])
        _, residuals = changed.response(point, candidate["controls"])
        require(
            dict(candidate["residuals"]) == residuals,
            "relief candidate physical residuals differ",
        )
    exact_equal(
        candidate["relief_exact" if single else "objective_exact"],
        value,
        "candidate relief objective differs",
    )
    require(
        lower <= value and data["upper"] == rounded(value, 1),
        "relief objective bounds contradict the candidate",
    )
    require(
        data["resolution"] in ("bounded", "minimum_verified"),
        "invalid relief resolution",
    )
    if data["resolution"] == "minimum_verified":
        require(
            q(data["upper"]) - q(data["lower"]) <= q(options.get("tolerance", 1e-8)),
            "relief minimum gap is not closed",
        )
