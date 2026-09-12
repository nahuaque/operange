"""Reconstruct physical responses, operating objectives and audit coverage."""

from fractions import Fraction as F
from math import ulp

from ._certificate_math import (
    UnsupportedCertificate,
    exact_equal,
    q,
    require,
    rounded,
    vector_equal,
)
from .domains import FiniteSet
from .contract_types import plain
from .primitives import BoxSet
from .objectives import LinearObjective


def evidence(result, name, method=None):
    found = next((e for e in result.evidence if e.evidence_id == name), None)
    require(found is not None, f"missing certificate: {name}")
    if method is not None:
        require(found.method == method, f"unexpected certificate method for {name}")
    return found.details


def evaluation(ctx, result):
    payload, point = result.payload, result.request["realization"]
    if payload.feasibility not in ("feasible", "infeasible"):
        raise UnsupportedCertificate(
            "No completed physical feasibility conclusion to verify."
        )
    require(ctx.membership(point), "realization is not a member of the declared domain")
    require(payload.membership.status == "inside", "membership conclusion differs")
    check = evidence(result, "membership", "declared_domain_membership")["check"]
    require(
        check["status"] == "inside", "membership evidence contradicts domain membership"
    )
    require(
        check["domain_ref"] == ctx.domain.ref.to_dict()
        and dict(check["realization"]) == dict(point),
        "membership certificate is for a different domain or realization",
    )
    require(
        tuple(payload.included_constraints)
        == tuple(c.constraint_id for c in ctx.included),
        "feasibility omits a selected requirement or equipment constraint",
    )
    if ctx.controller is None and payload.feasibility == "infeasible":
        ctx.system(point).contradiction(
            evidence(result, "recourse", "exact_bounded_linear_contradiction")
        )
        require(
            {v.quantity_ref: v.value for v in payload.values} == dict(point),
            "infeasibility certificate invents an operating response",
        )
        require(
            all(
                c.assessment == "unknown" and c.residual is None
                for c in payload.constraint_checks
            ),
            "joint infeasibility does not establish individual violations",
        )
        return
    commands = {
        v.quantity_ref: v.value
        for v in payload.values
        if v.quantity_ref in {c.name for c in ctx.model.controls}
    }
    if ctx.controller is not None:
        require(
            commands == ctx.commands(point),
            "stored commands differ from exact affine execution and rounding",
        )
        data = evidence(result, "controller", "exact_affine_command_evaluation")
        require(
            data["controller_ref"] == ctx.controller.ref.to_dict()
            and dict(data["commands"]) == commands
            and data["clipping"] is False,
            "controller execution certificate differs",
        )
        require(
            data["observations_by_control"]
            == {
                r.control: {n: point[n] for n in r.observes}
                for r in ctx.controller.rules
            },
            "controller observation evidence differs",
        )
    else:
        ctx.system(point).feasible(commands)
        if result.request.get("dispatch_source") == "shared_relief_candidate":
            require(
                dict(result.request["controls"]) == commands,
                "shared relief replay changed the proposed dispatch",
            )
    response = evidence(result, "response", "exact_physical_residual_check")
    if ctx.controller:
        require(
            response["controller_ref"] == ctx.controller.ref.to_dict(),
            "physical response names a different controller",
        )
    else:
        require(
            dict(response["fixed_controls"]) == ctx.system(point).fixed,
            "physical response changes fixed control declarations",
        )
    values, residuals = ctx.response(point, commands)
    if (
        ctx.controller is None
        and ctx.model.objective is not None
        and not isinstance(ctx.model.objective, LinearObjective)
    ):
        spec = ctx.model.objective
        values[spec.quantity_id] = rounded(
            sum(
                (
                    q(t.weight)
                    * ((q(commands[t.control]) - q(t.target)) / q(t.scale)) ** 2
                    for t in spec.targets
                ),
                F(0),
            )
        )
    require(
        {v.quantity_ref: v.value for v in payload.values} == values,
        "stored physical outputs or residuals differ from the declarations and commands",
    )
    require(
        {c.constraint_ref for c in payload.constraint_checks} == set(residuals),
        "constraint checks omit or add physical rows",
    )
    failed = []
    for spec in ctx.included:
        row = next(
            c
            for c in payload.constraint_checks
            if c.constraint_ref == spec.constraint_id
        )
        require(
            row.residual is not None
            and row.residual.value == residuals[spec.constraint_id],
            "constraint residual differs",
        )
        violated = q(residuals[spec.constraint_id]) > q(spec.tolerance)
        require(
            row.assessment == ("violated" if violated else "satisfied"),
            "constraint assessment differs",
        )
        if violated:
            failed.append(spec.constraint_id)
    require(
        payload.feasibility == ("infeasible" if failed else "feasible"),
        "physical feasibility conclusion differs",
    )
    if failed:
        data = evidence(result, "recourse", "fixed_controller_violation")
        require(
            ctx.controller is not None
            and data["controller_ref"] == ctx.controller.ref.to_dict()
            and dict(data["commands"]) == commands
            and tuple(data["violated_constraints"]) == tuple(failed),
            "controller failure certificate differs",
        )


def operating_objective(ctx, result):
    spec, obj = ctx.model.objective, result.payload.objective
    if (
        spec is None
        or ctx.controller is not None
        or result.payload.feasibility != "feasible"
    ):
        require(obj is None, "unexpected operating objective claim")
        return
    if obj is None:
        raise UnsupportedCertificate(
            "The feasible dispatch has no exported objective certificate."
        )
    data = evidence(result, "operating_objective", "exact_separable_lagrangian_bound")
    require(
        plain(data["declaration"]) == spec.to_dict(),
        "objective certificate changes the operating preference",
    )
    point = result.request["realization"]
    system = ctx.system(point)
    system.match(data, rhs="rhs_exact")
    base = {
        **point,
        **system.fixed,
        **{n: origin for n, origin, _ in system.coordinates},
    }
    commands = {
        v.quantity_ref: v.value
        for v in result.payload.values
        if v.quantity_ref in {c.name for c in ctx.model.controls}
    }
    if isinstance(spec, LinearObjective):
        sign = 1 if spec.sense == "minimize" else -1
        terms = {
            t.variable: q(t.coefficient) for t in ctx.model.output(spec.output).terms
        }
        constant = sign * ctx.output(spec.output, base)
        linear = [sign * terms.get(n, 0) * span for n, _, span in system.coordinates]
        diagonal = [F(0)] * len(linear)
        value, name, unit = (
            ctx.output(spec.output, {**point, **commands}),
            spec.output,
            ctx.model.output(spec.output).unit,
        )
    else:
        sign, constant = 1, F(0)
        linear, diagonal = (
            [F(0)] * len(system.coordinates),
            [F(0)] * len(system.coordinates),
        )
        for target in spec.targets:
            weight, difference = (
                q(target.weight) / q(target.scale) ** 2,
                q(base[target.control]) - q(target.target),
            )
            constant += weight * difference**2
            for i, (n, _, span) in enumerate(system.coordinates):
                if n == target.control:
                    linear[i], diagonal[i] = (
                        2 * weight * difference * span,
                        weight * span**2,
                    )
        value = sum(
            (
                q(t.weight) * ((q(commands[t.control]) - q(t.target)) / q(t.scale)) ** 2
                for t in spec.targets
            ),
            F(0),
        )
        name, unit = spec.quantity_id, "1"
    exact_equal(data["constant_exact"], constant, "objective constant differs")
    vector_equal(data["linear_exact"], linear, "objective linear terms differ")
    vector_equal(data["diagonal_exact"], diagonal, "objective quadratic terms differ")
    lower = system.lower_bound(constant, linear, diagonal, data["certificate"])
    require(
        lower <= sign * value, "objective lower bound exceeds the physical candidate"
    )
    lo, hi = (lower, value) if sign > 0 else (value, -lower)
    for field, v in (("value_exact", value), ("lower_exact", lo), ("upper_exact", hi)):
        exact_equal(data[field], v, f"{field} differs")
    require(
        obj.quantity_ref == name
        and obj.sense == ("minimize" if sign > 0 else "maximize")
        and data["sense"] == obj.sense,
        "objective identity or sense differs",
    )
    require(
        obj.attained_value == rounded(value)
        and obj.lower.value == rounded(lo, -1)
        and obj.upper.value == rounded(hi, 1)
        and obj.lower.unit == obj.upper.unit == unit,
        "reported objective values or units differ",
    )
    require(
        obj.attainment == "attained",
        "objective attainment differs from the physical candidate",
    )
    proof = next(e for e in result.evidence if e.evidence_id == "operating_objective")
    require(
        len(proof.measurements) == len(proof.tolerances) == 1
        and proof.measurements[0].name == "attained_value"
        and proof.measurements[0].value == rounded(value)
        and proof.measurements[0].unit == unit
        and proof.tolerances[0].name == "absolute_tolerance"
        and proof.tolerances[0].value == spec.tolerance
        and proof.tolerances[0].unit == unit,
        "objective measurements or tolerance differ",
    )
    closed = q(obj.upper.value) - q(obj.lower.value) <= q(spec.tolerance)
    if obj.optimality == "verified" or data["resolution"] in (
        "minimum_verified",
        "maximum_verified",
    ):
        require(closed, "objective optimality gap is not closed")
        require(
            data["resolution"]
            == ("minimum_verified" if sign > 0 else "maximum_verified"),
            "objective resolution contradicts its sense",
        )
    require(
        data["strictly_convex_on_adjustable_controls"]
        == (bool(diagonal) and all(d > 0 for d in diagonal)),
        "strict convexity claim differs",
    )


def coverage(ctx, result):
    payload = result.payload
    require(
        result.request["query"] == "audit" and payload.search is None,
        "not an audit certificate",
    )
    evaluations = {e.result_id: e for e in result.supporting_evaluations}
    require(
        set(evaluations) == {r.artifact_id for r in payload.evaluation_refs},
        "audit evaluation references differ",
    )
    if ctx.controller is not None and type(ctx.domain) is not FiniteSet:
        controller_enclosure(ctx, result)
    else:
        cases = ctx.generators()
        refs = payload.coverage.evaluated_support["scenarios"]
        require(len(refs) == len(cases), "audit omits declared scenarios or generators")
        for item, (name, point) in zip(refs, cases):
            require(item["name"] == name, "audit scenario identity differs")
            child = evaluations[item["evaluation_ref"]["artifact_id"]]
            require(
                item["evaluation_ref"] == child.ref.to_dict()
                and dict(child.request["realization"]) == point,
                "scenario references the wrong physical input",
            )
        require(
            {r["evaluation_ref"]["artifact_id"] for r in refs} == set(evaluations),
            "unexpected supporting evaluations",
        )
        if payload.coverage.method != "partial":
            expected = (
                "complete_finite"
                if type(ctx.domain) is FiniteSet
                else "analytical_domain"
            )
            require(
                payload.coverage.method == expected
                and all(e.execution == "completed" for e in evaluations.values()),
                "coverage claim is incomplete",
            )
            method = (
                "complete_finite_controller_replay"
                if ctx.controller is not None
                else "complete_finite_linear_recourse"
                if type(ctx.domain) is FiniteSet
                else "complete_vertex_linear_recourse"
            )
            evidence(result, "coverage", method)
            proof = next(e for e in result.evidence if e.evidence_id == "coverage")
            require(
                len(proof.measurements) == 1
                and proof.measurements[0].name == "scenario_count"
                and proof.measurements[0].unit == "count"
                and proof.measurements[0].value == len(cases),
                "coverage scenario count differs",
            )
        if payload.verdict == "pass":
            require(
                payload.coverage.method != "partial"
                and all(
                    e.payload.feasibility == "feasible" for e in evaluations.values()
                ),
                "passing coverage lacks feasible generators",
            )
    if payload.verdict == "fail":
        witness = payload.witness
        require(witness is not None, "failure has no witness")
        child = evaluations[witness.details["evaluation_ref"]["artifact_id"]]
        require(
            witness.details["evaluation_ref"] == child.ref.to_dict()
            and child.payload.feasibility == "infeasible"
            and witness.realizations == (child.request["realization"],),
            "failure witness differs from its infeasibility proof",
        )
        affected = (
            tuple(
                c.constraint_ref
                for c in child.payload.constraint_checks
                if c.assessment == "violated"
            )
            if ctx.controller
            else tuple(c.constraint_id for c in ctx.included if c.kind != "equation")
        )
        require(
            witness.affected_constraints == affected,
            "failure witness changes the affected constraints",
        )
        expected_kind = (
            "fixed_policy_failure"
            if ctx.controller or ctx.policy.mode == "fixed"
            else "individual_infeasibility"
        )
        require(
            witness.kind == expected_kind,
            "failure witness claims the wrong policy scope",
        )
        for name in witness.membership_evidence_refs + witness.recourse_evidence_refs:
            require(
                next(e for e in result.evidence if e.evidence_id == name)
                == next(e for e in child.evidence if e.evidence_id == name),
                "witness certificate differs from its supporting evaluation",
            )
        if ctx.controller is not None:
            require(
                evidence(result, "controller") == evidence(child, "controller"),
                "witness controller execution differs",
            )
    elif payload.verdict != "pass":
        raise UnsupportedCertificate("No conclusive audit verdict to verify.")


def controller_enclosure(ctx, result):
    if result.payload.coverage.method != "analytical_domain":
        raise UnsupportedCertificate("No complete controller enclosure to verify.")
    points = (
        [] if type(ctx.domain) is BoxSet else [point for _, point in ctx.generators()]
    )

    def maximum(offset, coefficients):
        if type(ctx.domain) is BoxSet:
            return offset + sum(
                (
                    a
                    * q(
                        ctx.domain.parameter(n).upper
                        if a > 0
                        else ctx.domain.parameter(n).lower
                    )
                    for n, a in coefficients.items()
                ),
                F(0),
            )
        return max(
            offset + sum((a * q(p[n]) for n, a in coefficients.items()), F(0))
            for p in points
        )

    rules, commands, errors = {}, {}, {}
    for r in ctx.controller.rules:
        coeff = {t.variable: q(t.coefficient) for t in r.terms}
        lo, hi = (
            -maximum(-q(r.offset), {n: -a for n, a in coeff.items()}),
            maximum(q(r.offset), coeff),
        )
        identity = (
            r.offset == 0 and len(r.terms) == 1 and abs(r.terms[0].coefficient) == 1
        )
        error = (
            F(0)
            if identity or not any(coeff.values())
            else q(ulp(rounded(max(abs(lo), abs(hi)), 1))) / 2
        )
        rules[r.control] = r
        errors[r.control] = error
        commands[r.control] = {
            "lower": rounded(lo),
            "upper": rounded(hi),
            "rounding_error_exact": str(error),
        }
    forms = {}
    for output in ctx.model.outputs:
        offset, coeff, error = q(output.offset), {}, F(0)
        for term in output.terms:
            a = q(term.coefficient)
            if term.variable in rules:
                rule = rules[term.variable]
                offset += a * q(rule.offset)
                error += abs(a) * errors[term.variable]
                for t in rule.terms:
                    coeff[t.variable] = coeff.get(t.variable, F(0)) + a * q(
                        t.coefficient
                    )
            else:
                coeff[term.variable] = coeff.get(term.variable, F(0)) + a
        rounded(maximum(offset, coeff) + error, 1)
        rounded(-maximum(-offset, {n: -a for n, a in coeff.items()}) - error, -1)
        forms[output.name] = offset, coeff, error
    bounds = {}
    for row in ctx.included:
        if row.kind == "equation":
            continue
        data = evidence(
            result, f"bound:{row.constraint_id}", "rounded_controller_affine_enclosure"
        )
        if data["control_bound"] is not None:
            name, side, limit = data["control_bound"]
            control = next(c for c in ctx.model.controls if c.name == name)
            require(
                row.constraint_id == f"control_{side}:{name}"
                and side in ("lower", "upper")
                and limit == getattr(control, side),
                "control enclosure changes a hard bound",
            )
            rule, sign = rules[name], -1 if side == "lower" else 1
            offset, coefficients, error = (
                sign * (q(rule.offset) - q(limit)),
                {t.variable: sign * q(t.coefficient) for t in rule.terms},
                errors[name],
            )
            upper = (
                q(limit) - q(commands[name]["lower"])
                if side == "lower"
                else q(commands[name]["upper"]) - q(limit)
            )
        else:
            limit = next(r for r in ctx.limits if r.name == row.constraint_id)
            sign = 1 if limit.relation == "le" else -1
            offset, coefficients, error = forms[limit.output]
            offset, coefficients = (
                sign * (offset - q(limit.limit)),
                {n: sign * a for n, a in coefficients.items()},
            )
            upper = maximum(offset, coefficients) + error
        exact_equal(data["offset_exact"], offset, "controller residual offset differs")
        require(
            {n: q(a) for n, a in data["coefficients_exact"].items()} == coefficients,
            "controller residual coefficients differ",
        )
        exact_equal(
            data["command_rounding_error_exact"],
            error,
            "command rounding enclosure differs",
        )
        stored = next(
            e for e in result.evidence if e.evidence_id == f"bound:{row.constraint_id}"
        )
        require(
            stored.measurements[0].value == rounded(upper, 1)
            and stored.measurements[0].unit
            == result.contract.quantity(row.residual_ref).unit,
            "controller residual upper bound differs",
        )
        bounds[row.constraint_id] = rounded(upper, 1)
    data = evidence(
        result, "enclosure", "exact_substitution_and_command_rounding_bounds"
    )
    require(
        data["controller_ref"] == ctx.controller.ref.to_dict()
        and data["command_bounds"] == commands,
        "controller command enclosure differs",
    )
    for proof in data["support_bounds"]:
        require(
            ctx.membership(proof["point"]), "support extremum lies outside the domain"
        )
        coefficients = {n: q(a) for n, a in proof["coefficients_exact"].items()}
        upper = maximum(q(proof["offset_exact"]), coefficients)
        exact_equal(proof["upper_exact"], upper, "affine support upper bound differs")
        exact_equal(
            q(proof["offset_exact"])
            + sum((a * q(proof["point"][n]) for n, a in coefficients.items()), F(0)),
            upper,
            "support point does not attain the bound",
        )
    require(
        evidence(result, "coverage", "rounded_controller_constraint_enclosures")[
            "requirement_bounds"
        ]
        == bounds,
        "coverage bounds differ",
    )
    require(
        result.payload.coverage.method == "analytical_domain",
        "controller coverage method differs",
    )
    require(
        [dict(p) for p in result.payload.coverage.evaluated_support["realizations"]]
        == [dict(e.request["realization"]) for e in result.supporting_evaluations],
        "controller coverage replay points differ",
    )
    if result.payload.verdict == "pass":
        require(
            all(
                bounds[c.constraint_id] <= c.tolerance
                for c in ctx.included
                if c.kind != "equation"
            ),
            "controller enclosure does not prove a pass",
        )
