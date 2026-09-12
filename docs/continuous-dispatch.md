# Audit adjustable dispatch over a continuous envelope

For a built-in `LinearProcessAdapter`, a continuous audit asks whether **every
admissible realization has some feasible permitted dispatch**. Controls can be
fixed, or selected after observing every process input at the single `operation`
stage. The model's affine coefficients, efficiencies, control bounds and
constraint tolerances remain fixed.

Two continuous domain declarations support this audit:

- `BoxSet`: check every physical corner, counting constant coordinates once.
- `ConvexHullSet(space, vertices, provenance=..., tolerance=1e-10)`: check every
  declared generator. `vertices` is a nonempty tuple of named `Scenario` records;
  redundant, duplicate and affinely dependent generators are permitted.

A `FiniteSet` still covers only its listed realizations. Constructing a hull is
an explicit engineering declaration that **all convex combinations** are also
admissible. It does not infer probabilities or an executable control rule.

## Compare unrestricted loads and scheduled operating modes

The [two-boiler model](linear-dispatch.md) has a 30 MW fuel supply. Its individual
peak modes require 18 MW of steam, but simultaneous peaks require 20 MW and
cannot be met. A hull of nominal and individual peak modes represents the
additional scheduling restriction `dryer + evaporator <= 18 MW` within this
example's load ranges.

```{literalinclude} ../examples/continuous_dispatch.py
:language: python
:start-at: def run_example(
:end-before: if __name__
```

| Study | Verdict | Coverage |
| --- | --- | --- |
| Dryer 10–12 MW, evaporator 6–8 MW | Fail | All four box corners checked; simultaneous peaks supply an infeasibility witness |
| Hull of nominal and individual peaks | Pass | Every convex combination of the three declared modes |
| Original box with fuel capacity increased to 32 MW | Pass | Entire original box, with the same service and permissions |

The scheduled hull changes the operating envelope. Increasing fuel capacity
instead restores the original envelope. Neither audit selects or prices the
equipment change.

```bash
uv run python -m examples.continuous_dispatch
uv run --extra cvxpy python -m examples.continuous_dispatch --backend cvxpy
```

## Why generators cover the entire domain

Suppose generator `u_i` has a checked feasible control vector `c_i`. For any
nonnegative weights summing to one, set:

```text
u = sum_i w_i * u_i
c = sum_i w_i * c_i
```

Every affine constraint residual at `(u, c)` is the same convex combination of
the generator residuals. It therefore stays within the same declared tolerance.
The control box is convex, and fixed commands remain fixed. Because adjustable
controls observe every input, there is no additional requirement to use the same
command at indistinguishable realizations. This establishes real-valued dispatch
existence throughout the hull.

Each generator dispatch is checked with exact rational arithmetic on the
declared physical floats. The audit preserves the original domain, operating
permissions, selected service requirements and all equipment limits in every
supporting evaluation. The coverage record lists generator names and evaluation
references; the `complete_vertex_linear_recourse` evidence records the theorem
and its assumptions. Passing coverage is `analytical_domain`, rather than
`complete_finite`.

A certified infeasible generator refutes the whole claim and can be replayed
with `claim.evaluate_result(audit.payload.witness.realizations[0])`. A verified
failure survives unresolved checks elsewhere. Without a failure, any unresolved
generator prevents a pass and yields partial coverage with an inconclusive
verdict. A solver's success or infeasibility status alone proves neither verdict.

## Enumeration and numerical limits

`claim.audit_result(max_vertices=256)` is the default. A box with `k` varying
scalar coordinates requires `2**k` checks; vector coordinates count by component.
A hull counts every declared generator, retaining duplicate labels. If the
required count exceeds the limit, the query returns unsupported with
`vertex_limit_exceeded` before evaluating any generator. Increase the explicit
limit to permit complete enumeration. There is no sampling fallback, and
`max_vertices` is not an option for finite-scenario audits.

SciPy remains the default. `backend="cvxpy"` reuses the existing prepared
dispatch programs as the generator inputs change. An optional operating
objective still selects each generator dispatch and retains its pointwise
optimality evidence. The continuous audit establishes feasibility only: it does
not certify interpolated objective values as optimal or optimize a domain-wide
objective.

The existence proof does not export a controller or certify how interpolated
commands behave after floating-point rounding. Audit an explicit
[`AffineController`](frozen-controllers.md) for that question. Partial-observation
adjustable policies, integer controls, uncertain control coefficients, general
polytopes, ellipsoids, intersections, unions and products remain unsupported
for this adjustable audit. Subclasses of the built-in adapter/domain types do
not automatically inherit this proof.

## Hull membership and portability

Hull membership needs neither nominal values nor normalization scales. Named
generators are members by construction. Other points require exact nonnegative
barycentric weights summing to one, or an exact separating certificate for an
outside verdict. Numerical LP proposals are checked independently; unresolved
membership stays `unknown`. The numerical `tolerance` does not widen the hull.
`contains()` raises when membership is unresolved.

With explicit coordinate nominals and scales, hulls also provide linear support:
the maximum is the largest generator value. This supports existing affine and
explicit-controller support audits. It does not add hulls to the convex distance
or intersection compilers.

`domain_from_manifest` and `domain_from_json` restore hull declarations.
Continuous audit results use the existing standard and compact result formats;
`result_from_json` validates their structure and identities without re-proving
the recorded mathematical evidence.
