"""Box support constraints for numerical affine-policy proposals.

Inputs use independent centered coordinates in [-1, 1]. A substituted affine
row has maximum center + sum(abs(slopes)); slopes sharing an input are combined
before taking absolute values. CVXPY compiles these inequalities to an LP.
The saved physical controller must still pass its separate exact audit.
"""

from fractions import Fraction as F

from .primitives import finite


def constraints(program, requirements, reserve):
    cp, model = program.cp, program.model
    centers = {n: (lo + hi) / 2 for n, (lo, hi) in program.input_ranges.items()}
    radii = {n: (hi - lo) / 2 for n, (lo, hi) in program.input_ranges.items()}
    commands = {}
    for n, (control, fixed, observes, variable) in program.rules.items():
        if fixed is not None:
            commands[n] = (cp.Constant(fixed), {})
            continue
        span = finite(float(F(control.upper) - F(control.lower)), "control span")
        center = control.lower + span * variable[0]
        slopes = {}
        for i, observed in enumerate(observes):
            offset, terms = program.observation_forms[observed]
            lo, hi = program.ranges[observed]
            midpoint = offset + sum((a * centers[k] for k, a in terms.items()), F(0))
            center += (
                span
                * variable[i + 1]
                * finite(float((midpoint - lo) / (hi - lo)), "observation center")
            )
            for k, a in terms.items():
                if radii[k]:
                    slopes[k] = slopes.get(k, 0) + span * variable[i + 1] * finite(
                        float(a * radii[k] / (hi - lo)), "observation slope"
                    )
        commands[n] = (center, slopes)

    def output_form(output):
        constant = F(output.offset)
        slopes, control_centers = {}, []
        for term in output.terms:
            n, a = term.variable, term.coefficient
            if n in commands:
                center, command_slopes = commands[n]
                control_centers.append(a * center)
                for observed, slope in command_slopes.items():
                    slopes[observed] = slopes.get(observed, 0) + a * slope
            else:
                constant += F(a) * centers[n]
                if radii[n]:
                    slope = finite(float(F(a) * radii[n]), "input slope")
                    slopes[n] = slopes.get(n, 0) + slope
        center = cp.Constant(finite(float(constant), "output constant")) + sum(
            control_centers, 0
        )
        return center, slopes

    def upper(form, *, sign=1, limit=0, scale=1):
        center, slopes = form
        radius = (
            cp.norm1(cp.hstack([slope / scale for slope in slopes.values()]))
            if slopes
            else 0
        )
        return sign * (center - limit) / scale + radius

    rows = []
    for control in model.controls:
        form = commands[control.name]
        rows.extend(
            (
                upper(form, sign=-1, limit=control.lower) <= 0,
                upper(form, limit=control.upper) <= 0,
            )
        )
    for row in model.operating_limits + tuple(
        r for r in model.requirements if r.name in requirements
    ):
        rows.append(
            upper(
                output_form(model.output(row.output)),
                sign=row.sign,
                limit=row.limit,
                scale=row.residual_scale,
            )
            <= (1 - reserve) * row.tolerance / row.residual_scale
        )
    spec = program.objective
    epigraph = cp.Variable(name="worst_case_cost")
    rows.append(
        upper(
            output_form(model.output(spec.output)),
            sign=1 if spec.sense == "minimize" else -1,
        )
        <= epigraph
    )
    return rows, epigraph
