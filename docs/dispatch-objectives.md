# Operating objectives for adjustable dispatch

An optional operating objective selects a preferred dispatch from the controls
that satisfy all equipment limits and selected service requirements. The
objective never trades off a violation of those requirements against fuel use
or command tracking. Omit it to retain feasibility-only dispatch.

## Minimize a declared affine output

Use an existing `AffineOutput` as the objective quantity. For the
[two-boiler example](linear-dispatch.md):

```python
from dataclasses import replace
from operange import LinearObjective

fuel_model = replace(model, objective=LinearObjective("fuel"))
claim = fuel_model.as_claim(loads)
result = claim.evaluate_result({"dryer": 10, "evaporator": 6})
objective = result.payload.objective
```

At 16 MW demand, minimum-fuel operation uses boiler A at 12 MW and boiler B
at approximately 4 MW, consuming approximately 23 MW of fuel. The exact
optimum accounts for the declared service tolerances. Units come from the
output declaration, and its affine expression may include an offset, inputs,
and controls. `LinearObjective("output_name", sense="maximize")` supports the
opposite preference. Weighted combinations can be declared as an affine output
with explicit coefficient units.

Linear objectives work with the default SciPy backend or `backend="cvxpy"`.
CVXPY reuses prepared problems when the matrix stays the same and loads or
objective coefficients change. The objective belongs to the model declaration
and contract identity; the numerical backend remains a query option.

## Stay close to target commands

For squared command tracking, declare each target in the control's physical
unit with an explicit positive scale and weight:

```python
from operange import ControlTarget, ControlTrackingObjective

tracking_model = replace(model, objective=ControlTrackingObjective((
    ControlTarget("boiler_a", target=8, scale=1, unit="MW"),
    ControlTarget("boiler_b", target=8, scale=1, unit="MW"),
)))
tracked = tracking_model.as_claim(loads).evaluate_result(
    {"dryer": 12, "evaporator": 6}, backend="cvxpy",
)
```

This minimizes the dimensionless quantity

```text
sum(weight * ((control - target) / scale)**2)
```

At 18 MW demand, equally weighted targets of 8 MW select approximately 9 MW
from each boiler. A larger weight or smaller scale makes departure from that
target more expensive in the objective. Target values can lie outside the
feasible range: they remain preferences, while hard limits determine what can
actually be dispatched. Duplicate targets, unknown controls, inconsistent
units, nonpositive scales or weights, and nonfinite declarations are rejected.

The default objective quantity is `dispatch_tracking_error`; `quantity_id`
can give it another unused name. Every positive-weight target contributes a
convex quadratic term. Targeting every adjustable control makes the objective
strictly convex and gives a unique mathematical minimizer on a nonempty feasible
set. Returned commands are floating-point approximations with checked objective
bounds; strict convexity does not promise identical command bits across solvers
or runs. Partial tracking and linear objectives can leave ties.

Tracking requires `backend="cvxpy"` and the optional `operange[cvxpy]` extra.
It uses a prepared convex quadratic program with Clarabel. A missing extra or
unsupported backend produces an explicit `unsupported` result.

## Read feasibility and optimality separately

For a verified feasible dispatch, `payload.objective` contains:

- `quantity_ref`, `sense`, and the value attained by the returned commands.
- Conservative `lower` and `upper` bounds on the mathematical optimum, in the
  objective's unit.
- `attainment="attained"` for the returned feasible candidate, and
  `optimality="verified"` only when the bounds close within the declared
  objective tolerance. Otherwise optimality is `unknown`.

Both objective declarations accept an absolute `tolerance` (default `1e-8`)
in the objective's unit. This is separate from solver tolerances and physical
constraint tolerances. `attainment="attained"` establishes an actual candidate
value; it does not by itself establish optimality.

If optimization fails to close its gap, the evaluator can fall back to a checked
feasible dispatch and report conservative objective bounds plus a diagnostic.
It may also repair a numerical boundary candidate using adjacent representable
commands or interpolation from a verified feasible dispatch. Every returned
command vector still passes exact checks against the original constraints.
If the objective value cannot be exported finitely, the physical result can
remain feasible with an objective diagnostic and no fabricated objective value.

An infeasible evaluation carries its checked contradiction and no objective or
invented control values. Finite audits retain scenario objective results, and
continuous box/hull audits retain generator objective results, but
their pass/fail verdict continues to concern physical feasibility. They neither
aggregate objectives across scenarios nor require optimality for a physical pass.

## Independently check the objective bounds

After substituting inputs and fixed controls, the minimization form is
`k + c.z + sum(d_i*z_i**2)`, with `d_i >= 0`, `A z <= b`, and `0 <= z <= 1`.
Maximization of an affine output is handled by negating that output. For any
nonnegative row multipliers `lambda`, a valid lower bound is:

```text
k - lambda.b
  + sum(min over 0 <= z_i <= 1 of
        d_i*z_i**2 + (c_i + (lambda.A)_i)*z_i)
```

Each one-dimensional minimum has an exact closed form. Operange computes it
using rationals of the original declared floats, independently of numerical
solver status or approximate stationarity. The zero-multiplier bound remains
available when no usable dual candidate is returned. The returned feasible
commands establish the other side of the objective interval. Bounds round
outward for export.

The `operating_objective` evidence records the declaration, exact polynomial,
row identities, multipliers, control transformation, box minimizers and bounds.
Standard and compact result exports use the existing `process_result/v1` schema.
Loading checks records and identities without rerunning the mathematical proof.

## Scope and replay

Objectives apply to fixed or fully observed single-stage dispatch over the
adapter's point evaluations, finite audits and continuous box/hull generator
evaluations. [Continuous audits](continuous-dispatch.md) prove feasibility
throughout the domain; they do not assert domain-wide optimality. Fixed controls
contribute their actual objective values and cannot be changed by the optimizer. Conflict
diagnosis and one-limit relief continue to answer their physical questions;
relief minimizes the named equipment change rather than the operating objective.

An explicit affine controller still executes its prescribed commands. Freezing
such a binding preserves the model's objective declaration but does not turn
controller replay into optimization or assert an optimum for those commands.
Objective changes appear as operating preferences in engineering comparisons.

This addition does not provide optimized-dispatch derivatives, lexicographic
tie-breaking, mixed-integer decisions, or general nonlinear objectives.

Run the complete example from a checkout:

```bash
uv run --extra cvxpy python -m examples.dispatch_objectives
```
