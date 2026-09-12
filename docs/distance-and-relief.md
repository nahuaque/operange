# Convex distances and joint equipment relief

Two extensions reuse the optional convex backend: threshold distances over
supported convex domains, and combinations of caller-selected operating-limit
changes. Install `operange[cvxpy]` and run the worked example:

```bash
uv run --extra cvxpy python -m examples.distance_and_relief
```

## Choose a physical distance metric

`NormalizedL2(space)` measures the Euclidean length of the scaled deviations:

```text
d(u) = sqrt(sum_i ((u_i - origin_i) / scale_i)**2)
```

Every coordinate has an explicit origin and positive scale. Units must align
with the domain; no unit conversion is inferred. Origins may be outside the
domain, and the metric's scales may differ from the domain normalization.
`measure()` returns a float rounded upward from the exact distance. Search
evidence retains the exact squared distance and a conservative square-root bound.
`NormalizedLInf` remains available when the largest individual scaled deviation
is the desired severity measure.

```python
claim = model.as_claim(domain, distance=NormalizedL2(domain.space))
boundary = claim.boundary_result(backend="cvxpy")
breaking = claim.breaking_result(
    backend="cvxpy",
    violation_margins={"header_capacity": 0.01},
)
```

The CVXPY distance backend supports `BoxSet`, `PolytopeSet`, `BudgetSet`,
`SimplexSet`, `EllipsoidSet`, and intersections or disjoint products of these
built-in convex sets. It reuses the [convex-domain compiler](convex-backends.md).
The query's explicit `backend="cvxpy"` enables compilation; an intersection does
not need a support backend for a fixed affine distance query. Every domain still
needs a verified feasible point and complete coordinate normalization.
Unions, finite scenario sets and custom domain implementations are unsupported.

The default `backend="scipy"` retains the existing L∞ search over boxes and
polytopes. Euclidean metrics and broader convex domains require the CVXPY backend;
missing dependencies return `unsupported`. A capability describes eligibility;
query options and returned evidence determine whether a particular search runs
and resolves.

The worked example intersects a load ellipsoid with a dryer cap of 11 MW. From
nominal loads of 10 and 6 MW, each scaled by 2 MW, it searches for a 0.01 MW
violation of an 18.7 MW header limit. The nearest point is approximately
`dryer=11`, `evaporator=7.71`. Its L∞ distance is 0.855; its Euclidean distance is
about 0.990467. Each selected requirement remains a separate target branch.

Saved affine controllers support these metrics too:

```python
claim = model.as_claim(domain, controller=controller, distance=NormalizedL2(domain.space))
frozen = claim.freeze()
reloaded = FrozenController.from_json(frozen.to_json()).as_claim()
result = reloaded.breaking_result(
    backend="cvxpy", constraints=("shared_fuel",),
    violation_margins={"shared_fuel": 0.01},
)
```

Controller distance bounds include command-rounding allowances. Candidates must
execute the actual saved rule and reach the physical target. The controller's
domain must also supply linear support to bound its commands and outputs; use
`Intersection(..., backend="cvxpy")` for that workflow. Frozen loading preserves
the metric without importing CVXPY or executing a solve.

## Check the distance independently

The program proposes a point, a nonnegative target multiplier and a vector for
the norm's dual ball. Operange bounds that vector's dual norm in exact arithmetic
and uses the existing factor-support decomposition to establish:

```text
min ||M z + d|| >= lambda*b + v.d - h_D(lambda*a - M.T*v)
```

Here `a.z >= b` is the target, `M z + d` is the metric displacement, and
`lambda >= 0` with `dual_norm(v) <= 1`. Proposed factor splits need not satisfy
an approximate dual equality: their exact residual is bounded over the finite
domain enclosure. Evidence exports the norm vector, multiplier, transforms,
factor supports and residual correction.

Upper bounds require exact membership in the original convex restrictions,
including equalities and ellipsoid quadratic forms, plus physical target replay.
A support upper bound below the target can prove unreachability. Solver status
alone cannot. The global lower bound combines every selected requirement branch;
a missing branch retains zero and cannot be dropped from a minimum claim.

Poor conditioning or a target with no accepted floating-point witness can leave
the result `bounded` or `unresolved`. These searches do not prove the whole claim
passes and do not add distance searches for freely adjustable dispatch.

## Declare a joint relief objective

A joint relief query selects distinct operating limits, each with a maximum
physical relaxation, a physical scale, a matching unit and an optional positive
weight (default 1):

```python
result = claim.evaluate_result(
    {"dryer": 12, "evaporator": 8}, backend="cvxpy",
    relief={
        "changes": [
            {"constraint": "boiler_a_capacity", "maximum": 2,
             "scale": 1, "unit": "MW", "weight": 1},
            {"constraint": "shared_fuel", "maximum": 2,
             "scale": 1, "unit": "MW", "weight": 1},
        ],
        "objective": "quadratic",
        "tolerance": 1e-8,
    },
)
relief = next(e.details for e in result.evidence if e.evidence_id == "relief")
```

| Objective | Dimensionless expression | Backend |
| --- | --- | --- |
| `linear` | `sum(weight * relief / scale)` | SciPy or CVXPY |
| `quadratic` | `sum(weight * (relief / scale)**2)` | CVXPY |

Scales and weights define the comparison, including between limits with different
units. They are caller assumptions, not inferred prices or an investment ranking.
Service requirements, fixed commands, hard control bounds and unselected limits
remain in force. Upper operating limits relax upward; lower limits relax downward.
To study an equipment capacity, declare it as an operating limit within a control
box that covers the contemplated changes. Relief does not enlarge the hard box.

In the example, the linear objective selects 1 MW of fuel relief. The quadratic
objective selects approximately 0.48 MW of boiler A relief and 0.64 MW of fuel
relief, giving an objective of 0.64. The implementation augments the existing
bounded linear system and reuses the dispatch objective solver and exact
separable linear/quadratic lower-bound checks.

The `relief` evidence reports dimensionless objective bounds, the declaration,
its certificate, and any candidate's physical changes, dispatch and residuals.
`minimum_verified` means those bounds close within the objective tolerance.
`unreachable` requires an exact contradiction even with the permitted changes.
`bounded` or `unresolved` preserves numerical limitations. The original evaluation
still reports feasibility under the original equipment limits.

Apply `candidate.changes[*].changed_limit` to a new model and re-audit the full
declared domain. The example does this separately for both objectives. A relief
minimum at one realization does not establish restoration elsewhere. Existing
one-limit queries retain their API and physical-unit bounds.

For one plan shared by the full domain, use
`claim.audit_result(relief={"changes": ..., "objective": ...})` instead.
[Shared equipment relief](shared-relief.md) supports finite scenarios and
continuous boxes/hulls, and includes a separate full-domain candidate re-audit.
