"""Independent local-value checks from physical declarations, without a solver."""

from fractions import Fraction as F

from ._certificate_math import Context, UnsupportedCertificate, dot, q, require
from .contract_types import plain
from .dispatch_sensitivity import DispatchChange
from .objectives import LinearObjective


def _physical_rows(ctx, point, change):
    """Rebuild physical inequalities independently of the producer's LP compiler."""
    controls = [c.name for c in ctx.model.controls]
    rows, rhs, delta, refs = [], [], [], []
    operating = {r.name for r in ctx.model.operating_limits}
    for limit in ctx.limits:
        output = ctx.model.output(limit.output)
        terms = {t.variable: q(t.coefficient) for t in output.terms}
        sign = 1 if limit.relation == "le" else -1
        constant = q(output.offset) + sum(
            (a * q(point[n]) for n, a in terms.items() if n in point), F(0)
        )
        rows.append([sign * terms.get(n, F(0)) for n in controls])
        rhs.append(q(limit.tolerance) + sign * (q(limit.limit) - constant))
        refs.append(
            f"{'operating_limit' if limit.name in operating else 'requirement'}:{limit.name}"
        )
        delta.append(
            -sign * terms.get(change.name, F(0))
            if change.kind == "input"
            else F(
                sign
                if change.kind == "operating_limit" and change.name == limit.name
                else 0
            )
        )
    fixed = {
        r.control: r.fixed_value for r in ctx.policy.rules if r.fixed_value is not None
    }
    for i, c in enumerate(ctx.model.controls):
        for side, sign in (("lower", -1), ("upper", 1)):
            rows.append([F(sign if i == j else 0) for j in range(len(controls))])
            rhs.append(sign * q(getattr(c, side)))
            delta.append(
                F(
                    sign
                    if change.kind == f"control_{side}" and change.name == c.name
                    else 0
                )
            )
            refs.append(f"control_{side}:{c.name}")
            if c.name in fixed:
                rows.append(rows[-1])
                rhs.append(sign * q(fixed[c.name]))
                delta.append(F(0))
                refs.append(f"fixed_{side}:{c.name}")
    objective = ctx.model.output(ctx.model.objective.output)
    sign = 1 if ctx.model.objective.sense == "minimize" else -1
    terms = {t.variable: q(t.coefficient) for t in objective.terms}
    constant = sign * (
        q(objective.offset)
        + sum((v * q(point[n]) for n, v in terms.items() if n in point), F(0))
    )
    constant_delta = (
        sign * terms.get(change.name, F(0)) if change.kind == "input" else F(0)
    )
    cost = [sign * terms.get(n, F(0)) for n in controls]
    return controls, rows, rhs, delta, refs, cost, constant, constant_delta, sign


def _check_side(ctx, point, change, side):
    names, rows, rhs, delta, refs, cost, constant, constant_delta, sign = (
        _physical_rows(ctx, point, change)
    )
    expected = {"row_refs", "multipliers_exact"}
    if side.status == "verified":
        expected |= {"controls_exact", "control_rates_exact", "base_value_exact"}
    require(set(side.proof) == expected, "unexpected local certificate fields")
    require(
        tuple(side.proof["row_refs"]) == tuple(refs),
        "local proof omits or substitutes physical rows",
    )
    dual = [q(v) for v in side.proof["multipliers_exact"]]
    require(
        len(dual) == len(rows) and all(v >= 0 for v in dual),
        "invalid local dual multipliers",
    )
    direction = 1 if side.side == "forward" else -1
    require(
        side.valid_step is not None and 0 < q(side.valid_step) <= q(change.step),
        "invalid local validity interval",
    )
    if side.status == "infeasible":
        require(
            all(dot([row[j] for row in rows], dual) == 0 for j in range(len(names))),
            "direction exclusion has nonzero variable coefficients",
        )
        require(
            dot(rhs, dual) == 0 and direction * dot(delta, dual) < 0,
            "ray does not exclude every positive directional change",
        )
        return
    require(
        set(side.proof["controls_exact"])
        == set(side.proof["control_rates_exact"])
        == set(names),
        "local proof changes control coordinates",
    )
    x = [q(side.proof["controls_exact"][n]) for n in names]
    dx = [q(side.proof["control_rates_exact"][n]) for n in names]
    for row, b, h in zip(rows, rhs, delta):
        require(dot(row, x) <= b, "local base violates a physical constraint")
        end = direction * q(side.valid_step)
        require(
            dot(row, [a + end * rate for a, rate in zip(x, dx)]) <= b + end * h,
            "local interval extends beyond physical feasibility",
        )
    require(
        all(c + dot([row[j] for row in rows], dual) == 0 for j, c in enumerate(cost)),
        "local certificate violates exact stationarity",
    )
    value, slope = constant + dot(cost, x), constant_delta + dot(cost, dx)
    require(value == constant - dot(rhs, dual), "base primal/dual gap is not zero")
    require(
        slope == constant_delta - dot(delta, dual),
        "primal and dual slope certificates disagree",
    )
    require(
        q(side.proof["base_value_exact"]) == sign * value,
        "local base objective has the wrong value or sense",
    )
    require(
        q(side.slope_exact) == sign * slope,
        "reported derivative differs from the physical proof",
    )


def verify_dispatch_sensitivity(verifier, result):
    def declarations():
        if result.execution in ("unsupported", "invalid") or not result.sensitivities:
            raise UnsupportedCertificate(
                "This sensitivity query asserts no local certificates."
            )
        ctx = Context(result.contract)
        require(
            ctx.controller is None and type(ctx.model.objective) is LinearObjective,
            "local value proof requires a linear objective and no saved controller",
        )
        request = result.request
        require(
            set(request)
            == {
                "query",
                "realization",
                "changes",
                "backend",
                "max_refinements",
                "scope",
            },
            "unexpected dispatch sensitivity request fields",
        )
        require(
            request["query"] == "dispatch_sensitivity"
            and request["scope"]
            == "ambient objective-value derivatives of real-valued optimized dispatch; one coordinate at a time",
            "unsupported sensitivity semantics",
        )
        require(
            request["backend"] in ("scipy", "cvxpy")
            and type(request["max_refinements"]) is int
            and 0 <= request["max_refinements"] <= 32,
            "invalid numerical search declaration",
        )
        require(
            1 <= len(ctx.model.controls) <= 64
            and len(ctx.limits) + 4 * len(ctx.model.controls) <= 4096,
            "local proof exceeds the declared size limit",
        )
        changes = tuple(DispatchChange(**plain(c)) for c in request["changes"])
        require(
            1 <= len(changes) <= 64
            and len({c.change_id for c in changes}) == len(changes),
            "invalid sensitivity coordinate selection",
        )
        require(
            tuple(s.change for s in result.sensitivities) == changes,
            "local report omits, reorders or substitutes requested changes",
        )
        point = ctx.domain.space.validate(request["realization"])
        require(
            ctx.membership(point), "sensitivity base is outside the declared domain"
        )
        require(
            result.base is not None
            and result.base.request.get("realization") == point
            and result.base.payload.feasibility == "feasible",
            "sensitivity has no matching feasible base",
        )
        for change, sensitivity in zip(changes, result.sensitivities):
            if change.kind == "input":
                unit = ctx.domain.space.coordinate(change.name).unit
            elif change.kind == "operating_limit":
                limit = next(
                    r for r in ctx.model.operating_limits if r.name == change.name
                )
                unit = ctx.model.output(limit.output).unit
            else:
                unit = next(c.unit for c in ctx.model.controls if c.name == change.name)
            require(
                change.unit == unit, "perturbation unit differs from the declaration"
            )
            for side in (sensitivity.backward, sensitivity.forward):
                require(
                    side.unit
                    == f"{ctx.model.output(ctx.model.objective.output).unit}/{unit}",
                    "incorrect local derivative unit",
                )
        return ctx, point

    context = verifier.check("dispatch_sensitivity/declarations", declarations)
    if context is None:
        return
    ctx, point = context
    verifier.result(result.base, "dispatch_sensitivity/base")
    asserted = False
    for i, sensitivity in enumerate(result.sensitivities):
        for side in (sensitivity.backward, sensitivity.forward):
            if side.status == "unresolved":
                continue
            asserted = True
            verifier.check(
                f"dispatch_sensitivity/{i}/{side.side}",
                lambda s=side, c=sensitivity.change: _check_side(ctx, point, c, s),
            )
    if not asserted:
        raise UnsupportedCertificate("No direction has a local certificate to verify.")
