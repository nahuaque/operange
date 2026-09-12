# Linear models with adjustable controls

`LinearProcessAdapter` checks whether bounded controls can jointly satisfy
equipment limits and selected engineering requirements at each declared
realization. Its first audit scope is an explicit `FiniteSet`, with controls
fixed or chosen after all inputs are observed. The physical model remains
linear: coefficients, offsets, control bounds and efficiencies are constant.

## Declare two boilers sharing a fuel supply

This example extends the shared-steam study with two boilers. Boiler A can
deliver 12 MW at 80% efficiency, and B can deliver 10 MW at 50% efficiency.
The fuel supply is limited to 30 MW. Demand has four synthetic operating cases.

```{literalinclude} ../examples/linear_dispatch.py
:language: python
:start-at: from operange import (
:end-before: def run_example():
```

`LinearControl` bounds are finite and use the declared physical units.
`AffineOutput` expressions may reference inputs and controls, with explicit
coefficient units. Unit labels do not perform conversion. Outputs cannot
reference other outputs; expand the linear expression explicitly.

Both service requirements and equipment limits use `AffineRequirement` records.
Their roles differ: `operating_limits` always apply, while a claim can select a
subset of `requirements`. Here `meet_load` requires delivery at least equal to
demand, and `no_excess_steam` prevents excess delivery. Together they express
the steam balance within the two declared tolerances. A coupled fuel limit
applies to the sum of both boilers' consumption.

## Compare fixed commands with adjustment

```python
model, loads = example()
adjustable = model.as_claim(loads)
nominal = adjustable.evaluate_result({"dryer": 10, "evaporator": 6})
audit = adjustable.audit_result()

fixed = RecoursePolicy("fixed", (
    DecisionRule("boiler_a", "MW", "operation", fixed_value=10),
    DecisionRule("boiler_b", "MW", "operation", fixed_value=6),
))
fixed_audit = model.as_claim(loads, recourse=fixed).audit_result()
```

The default policy gives each control the `operation` stage and every input
as an observation. The model solves each realization independently. Explicit
static policies can mix fixed and adjustable controls, but each adjustable
control must observe all inputs. Fixed commands remain subject to equipment
bounds. Partial observations, multiple stages and shared unobserved decisions
are unsupported; they require a different policy-feasibility formulation.

| Case | Demand | Fixed A=10, B=6 | Adjustable operation |
| --- | --- | --- | --- |
| Nominal | 16 MW | Feasible | Feasible |
| Dryer peak | 18 MW | Infeasible | Feasible |
| Evaporator peak | 18 MW | Infeasible | Feasible |
| Combined peaks | 20 MW | Infeasible | Infeasible |

At 20 MW demand, even using A at its maximum 12 MW requires another 8 MW
from B. Fuel consumption is then `1.25 * 12 + 2 * 8 = 31 MW`, above the shared
30 MW supply. Merely adding boiler B capacity would not resolve that limitation.

Feasible evaluations return controls, outputs and checked constraint residuals
in `payload.values` and `payload.constraint_checks`. This is a feasibility
query: it does not minimize fuel, select an economic optimum or promise a
unique dispatch. The numerical phase-I objective only helps find a feasible
candidate and is not a process objective in the result.

## Re-audit an equipment change

```python
from dataclasses import replace

enlarged = replace(model, operating_limits=tuple(
    replace(limit, limit=32) if limit.name == "shared_fuel" else limit
    for limit in model.operating_limits
))
repaired = enlarged.as_claim(loads).audit_result()
```

The 32 MW fuel supply passes all four original scenarios with the same service
requirements and observation permissions. The example exports both studies:

```bash
uv run python -m examples.linear_dispatch
```

## What the numerical evidence establishes

A pass requires a physical control vector whose constraints are independently
checked using exact rational arithmetic on the declared floating-point values.
The check includes rounding incurred when solver coordinates are converted
back to physical controls. Physical control bounds have zero tolerance;
each service or equipment row uses its own declared tolerance. Exported signed
residuals round upward, preserving conservative comparisons.

For infeasibility, the adapter substitutes fixed inputs and controls, transforms
each remaining bounded control to `0 <= z <= 1`, and forms inequalities
`A z <= b`. Each row includes its declared tolerance and residual scale.
For any nonnegative multipliers `lambda`, feasibility would require:

```text
min over the control box of (lambda A) z <= lambda b
```

The left side is the sum of the negative weighted coefficients, since each
coordinate ranges from zero to one. If that minimum is strictly greater than
the right side, the constraints are incompatible for every bounded control
choice. The adapter verifies this strict inequality with exact rationals.
Single-row contradictions are checked directly; coupled contradictions use
solver-proposed multipliers, independently rechecked against the exact rows.
No approximate dual equality or solver infeasibility status is trusted.

The `recourse` evidence records the row identities, exact coefficients and
right-hand sides, nonnegative multipliers, control transformation and positive
contradiction gap. An infeasible result reports joint infeasibility, without
inventing control values, individual violations or a minimal conflict set.

If neither a feasible physical candidate nor a contradiction is verified,
the evaluation is unresolved. Poor conditioning or a feasible set with no
representable feasible floating-point control vector can therefore remain
unresolved. `solver_tolerance` controls numerical candidate generation; it does
not relax the physical bounds or declared constraint tolerances.

## Coverage, replay and limits

A passing audit covers only the listed finite realizations. Nominal values or
input normalization scales are unnecessary. Scenarios with identical physical
realizations retain their separate names in the coverage record.
If some scenarios remain unresolved, coverage is partial. A verified failure
in another scenario remains a valid witness and the verdict can still be `fail`.

Replay a failed realization with `claim.evaluate_result(witness.realizations[0])`.
Portable results use the existing `process_result/v1` schema and support
`to_json(compact=True)`. Loading checks records and identities; it does not rerun
the model or reverify the mathematical certificate. Model declarations can be
explicitly reconstructed using `LinearProcessAdapter(**model.to_dict())`.

Point evaluation is also available for other aligned domains when membership
is established. Continuous-domain recourse audits, dispatch derivatives,
closest-failure searches, nonlinear or integer controls, and executable causal
controllers are outside this adapter's current scope.

## Diagnose a conflict and quantify one-limit relief

Point evaluations can request additional evidence without changing the original
physical verdict:

```python
diagnosis = model.as_claim(loads).evaluate_result(
    {"dryer": 12, "evaporator": 8},
    diagnose=True,
    relief={"constraint": "shared_fuel", "maximum": 2, "tolerance": 1e-8},
)
conflict = next(e.details for e in diagnosis.evidence if e.evidence_id == "conflict")
relief = next(e.details for e in diagnosis.evidence if e.evidence_id == "relief")
```

The conflict query removes rows only after verifying that the remaining rows
are still infeasible. `irreducible=True` requires a verified feasible dispatch
for deleting each retained row. It means an irreducible row conflict **relative
to the declared control bounds and fixed commands**, not a minimum-cardinality
conflict across all equipment assumptions. Unresolved deletion checks preserve
the known contradiction and report `irreducible=False`.

For the boilers, the retained rows are `meet_load` and `shared_fuel`.
`certificate_bound_refs` identifies `control_upper:boiler_a` as a bound used
by the contradiction. The evidence preserves all background control bounds,
the exact contradiction and feasible row-removal dispatches. No control vector
is fabricated for the original infeasible evaluation.

The relief query relaxes one named **operating limit** in its permitted direction,
up to the caller's `maximum`, in that output's physical unit. It preserves
service requirements and operating permissions. It reports a verified lower
bound, an independently checked changed-limit candidate when available, and
`minimum_verified`, `bounded`, `unreachable` or `unresolved`. A `ge` limit is
relaxed downward; a `le` limit is relaxed upward. `unreachable` requires a
checked contradiction even at the maximum allowed relief. Feasible cases can
return zero relief. Numerical solver status alone proves none of these results.

Here the minimum extra fuel capacity is approximately 1 MW. More precisely,
the mathematical minimum is `1 - 3e-8` MW after accounting for this example's
declared service and fuel tolerances; the returned float limit is checked independently.
The bounds can retain a small gap when representable commands or limits prevent
attainment of the mathematical optimum. Inspect `resolution` and both bounds.

This is a pointwise adjustable-dispatch diagnosis. It does not establish that
the same change restores a saved controller or the full uncertainty domain.
Construct a candidate claim and use `compare_changes` to re-audit that question.
The [continuous controller example](frozen-controllers.md) demonstrates this
complete workflow. Both diagnostic records use the existing standard and compact
`EvaluationResult` exports; loading validates records and identities, without
re-proving the mathematical evidence.
