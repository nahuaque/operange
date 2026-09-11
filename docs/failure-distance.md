# Distance to a limit or a positive violation

For a fixed affine model, Operange can find the nearest admissible condition
that reaches a declared requirement limit or exceeds it by a specified amount.
The supported domains are `BoxSet` and `PolytopeSet`; distance is an explicitly
declared `NormalizedLInf` metric. Controls stay fixed throughout the search.

## Define a distance and two distinct targets

The shared utility example has nominal steam demand of 16 MW and a 19 MW
header limit. Both process loads can increase by 2 MW. It also has a 24 MW
fuel limit at fixed 80% boiler efficiency.

```{literalinclude} ../examples/failure_distance.py
:language: python
:start-at: from operange import (
:end-before: def run_example():
```

```python
model, loads = example()
claim = model.as_claim(loads, distance=NormalizedLInf(loads.space))

boundary = claim.boundary_result()
breaking = claim.breaking_result(
    violation_margins={"steam_header": 0.01, "boiler_fuel": 0.01},
)
```

For each coordinate, this metric divides the absolute change from the declared
origin by its scale, then takes the largest value. With 2 MW scales, simultaneous
1.5 MW increases have distance 0.75. Distance origins and scales belong to the
distance object; they need not match the domain's normalization. An origin may
lie outside the domain, but every returned candidate must lie inside it.

| Query | Target | Minimum distance | Governing requirement |
| --- | --- | --- | --- |
| Boundary | Signed residual at least zero | 0.75 | Steam header |
| Positive violation | Signed residual at least 0.01 MW | 0.7525 | Steam header |
| Fuel boundary alone | Fuel reaches 24 MW | 0.8 | Boiler fuel |

The boundary occurs at dryer demand 11.5 MW and evaporator demand 7.5 MW.
It reaches the engineering limit without violating the accepted requirement.
A `breaking_result` requires a positive violation amount for **every selected
requirement**, expressed in that output's physical units and strictly greater
than its declared tolerance. The amount is measured from the engineering limit;
it is not added to the tolerance. No unit conversion is inferred.

To search only one requirement, select it in the claim:

```python
fuel_claim = model.as_claim(
    loads, requirements=("boiler_fuel",), distance=NormalizedLInf(loads.space),
)
fuel_boundary = fuel_claim.boundary_result()
```

The search uses the union of the selected threshold sets: reaching **any one**
is sufficient. It does not require simultaneous violation of every requirement.
If the origin already reaches a target, the distance is zero. If it is already
beyond a requirement's tolerance, even a boundary query can carry a true failure
witness. Boundary attainment alone is never promoted to failure.

## Coupled restrictions change the answer

```python
restricted = PolytopeSet(loads, (
    LinearConstraint("dryer_cap", {"dryer": 1}, 0.5),
))
constrained = model.as_claim(restricted, distance=NormalizedLInf(loads.space))
restricted_boundary = constrained.boundary_result()
restricted_breaking = constrained.breaking_result(
    violation_margins={"steam_header": 0.01, "boiler_fuel": 0.01},
)
```

The restriction caps the dryer's normalized deviation at 0.5, or demand at
11 MW. The steam boundary moves to dryer 11 MW and evaporator 8 MW, distance
1.0. Neither selected 0.01 MW violation is reachable in this restricted domain.
This restriction changes the uncertainty assumption; it is not an equipment
repair. Polytope equality constraints apply in both directions, with no widening
by their numerical classification tolerance.

Run all four studies and export their results:

```bash
uv run python -m examples.failure_distance
```

## Read the bounds and the engineering verdict separately

```python
search = breaking.payload.search
print(search.resolution, search.lower.value, search.upper.value)
print(search.severity["reported_threshold_requirement"])
replay = claim.evaluate_result(breaking.payload.witness.realizations[0])
```

The search resolutions have these meanings:

| Resolution | Evidence |
| --- | --- |
| `minimum_verified` | An admissible evaluated candidate reaches a target, and its upper distance bound closes against the global lower bound within `distance_tolerance`. |
| `bounded` | An admissible candidate exists, but the lower/upper gap has not closed. |
| `unreachable` | Every selected target branch is independently proved infeasible in the domain. |
| `unresolved` | No target candidate is verified and unreachability has not been established. |

`distance_tolerance` defaults to `1e-8` in dimensionless distance units. It
controls the accepted optimality gap; it never relaxes domain membership or a
physical violation target. Lower bounds round downward and candidate distances
round upward for export. A verified minimum is therefore a bounded numerical
minimum with the stated gap, not a claim that the exported decimal is exact.

`Search.threshold` reports the governing candidate branch's physical threshold,
or the first selected requirement when there is no candidate. The full target
union is recorded in `Search.severity.thresholds`, including each residual
quantity and physical unit. Ties may have more than one closest requirement;
the representative is chosen deterministically, with all branches retained in
evidence.

A verified positive violation gives `payload.verdict == "fail"`, even if the
minimum distance remains only bounded. A resolved boundary without a violated
requirement gives `not_assessed`. An unreachable positive target also gives
`not_assessed`: smaller violations could still exist. Use `audit_result()` for
the full claim verdict. Threshold results retain partial point-evaluation
coverage rather than claiming that a domain audit was performed.

## How the certificates work

Each selected requirement becomes one linear program. Its variables are domain
normalized inputs and a nonnegative distance radius. Rows enforce the polytope,
the absolute-distance inequalities and the physical residual target. Exact finite
variable bounds come from the enclosing box and the distance metric.

For a minimization with objective `c v`, rows `A v <= b`, and a finite box `B`,
every nonnegative multiplier vector gives the lower bound:

```text
min over v in B of (c + lambda A) v - lambda b <= minimum distance
```

The box minimum is the sum of the smaller endpoint contribution for each
coefficient. The adapter checks the expression using exact rationals of the
declared floats. It remains valid when the solver's proposed dual equality has
a residual. Similarly, a strictly positive value of
`min_B (lambda A) v - lambda b` proves a branch unreachable.

Candidate points are converted back to physical units and checked against the
exact domain constraints and exact threshold. The ordinary membership and affine
evaluation must also succeed. Small rounding repairs are only proposed candidates;
they do not receive a relaxed membership test. Solver status, reported objective
values and approximate feasibility flags are never sufficient by themselves.

Every branch records its exact rows, bounds, objective, proposed multipliers,
checked lower bound or contradiction, and solver attempts. The global lower
bound is the minimum over all branches not proved unreachable. An unresolved
branch remains in that calculation, preventing a false global minimum.

## Scope and portability

Poor conditioning, unrepresentable distances or a target face without an
accepted floating-point candidate can leave a search unresolved or bounded.
The result preserves any verified evidence. This implementation adds no distance
search for adjustable controls, nonlinear models, finite domains, budgets,
ellipsoids, simplexes or composite sets, even when those domains support audits.

Results use the existing `process_result/v1` format, including compact transport:

```python
restored = result_from_json(breaking.to_json(compact=True))
```

Import `result_from_json` from `operange`. Loading validates records and identities;
it does not run the model or reprove the mathematical certificate.
