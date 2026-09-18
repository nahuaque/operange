"""Box support constraints for numerical affine-policy proposals.

Inputs use independent centered coordinates in [-1, 1]. A substituted affine
row has maximum center + sum(abs(slopes)); slopes sharing an input are combined
before taking absolute values. CVXPY compiles these inequalities to an LP.
The saved physical controller must still pass its separate exact audit.
"""

from fractions import Fraction as F

from .primitives import finite
from ._linear_problem import bound_rows


def constraints(program, reserve):
    cp = program.cp
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

    def output_form(expression):
        constant = expression.constant
        slopes, control_centers = {}, []
        for n, exact_coefficient in expression.terms:
            a = finite(float(exact_coefficient), "affine coefficient")
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

    def upper(form, *, scale=1):
        center, slopes = form
        radius = (
            cp.norm1(cp.hstack([slope / scale for slope in slopes.values()]))
            if slopes
            else 0
        )
        return center / scale + radius

    physical = program.physical
    declarations = (
        tuple(row for c in physical.variables for row in bound_rows(c)) + physical.rows
    )
    rows = [
        upper(
            output_form(row.expression),
            scale=finite(float(row.source.scale), "row scale"),
        )
        <= finite(
            float((1 - F(reserve)) * row.tolerance / row.source.scale),
            "reserved tolerance",
        )
        for row in declarations
    ]
    epigraph = cp.Variable(name="worst_case_cost")
    rows.append(upper(output_form(physical.objective.expression)) <= epigraph)
    return rows, epigraph
