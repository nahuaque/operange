# Vector parameters and uncertainty geometries

Implemented on the experimental branch, 2026-09-11. This extends the
[shared DSL](dsl-primitives.md) with labeled vectors, four continuous geometries,
and typed composition. These declarations describe admissible process inputs.
They do not execute process models or add presentation or business logic.

## Coordinates and vector parameters

`VectorParameter` declares one labeled axis, a common physical unit, nominal
component values, finite bounds, positive normalization scales, and provenance.
Nominal values must be label-keyed mappings. Bounds and scales accept either
label-keyed mappings or scalar broadcasts. Arbitrary tensors remain future work.

```python
from operange import (
    Axis, BoxSet, Intersection, SimplexSet, VectorParameter,
)

feed = VectorParameter(
    name="feed", unit="mole_fraction",
    axis=Axis("species", ("methane", "carbon_dioxide")),
    nominal={"methane": 0.75, "carbon_dioxide": 0.25},
    lower={"methane": 0.5, "carbon_dioxide": 0.1},
    upper={"methane": 0.9, "carbon_dioxide": 0.5},
    scale=0.25,
    provenance="Synthetic feed specification",
)
envelope = BoxSet((feed,))
fractions = SimplexSet(feed.space, basis="mole")
domain = Intersection((envelope, fractions))
point = domain.space.flatten({
    "feed": {"carbon_dioxide": 0.2, "methane": 0.8},
})
assert domain.membership(point).status == "inside"
assert domain.space.unflatten(point)["feed"]["methane"] == 0.8
```

Public geometry operations accept flat, named scalar coordinates. Use
`ParameterSpace.flatten`/`unflatten` at a nested application boundary, or
`feed.flatten`/`unflatten` for just that vector. `feed.name_for("methane")`
returns its canonical scalar coordinate name. Component names encode the
parameter, axis and label without delimiter collisions; callers should use
the helper instead of constructing them. Names, units, axis identities and
label sets determine alignment, independently of positional order. Missing,
extra, or duplicate labels are input errors. No unit conversion is inferred.

`ParameterSpace` holds coordinate metadata and normalization, not bounds.
For a coordinate with physical value `u`, nominal `n`, and scale `s`, the
normalized coordinate is `z = (u - n) / s`. All new geometries require complete
nominals and scales. A vector's scalar components retain their vector metadata
in the space and manifest. Scalar `Parameter` and existing scalar box manifests
remain supported.

## Membership and operations

| Domain | Admissible values | Operations implemented |
| --- | --- | --- |
| `BoxSet` | Inclusive physical bounds for scalar/vector components | Membership, separable linear support |
| `FiniteSet` | Explicit labeled physical realizations | Membership, enumeration; linear support with explicit normalization |
| `PolytopeSet` | Finite box intersected with named linear inequalities/equalities in `z` | Membership, checked linear support bounds |
| `SimplexSet` | Nonnegative physical fractions summing to one | Membership, analytical linear support |
| `BudgetSet` | Box intersected with `abs(z_i) <= 1` and `sum(abs(z_i)) <= budget` | Membership, analytical linear support |
| `EllipsoidSet` | `z.T @ inverse(shape) @ z <= radius**2` | Membership, analytical linear support |
| `Product`, `Union` | Typed composition described below | Membership; linear support when every factor supplies normalized support |
| `Intersection` | Shared convex restrictions | Membership; optional checked support with `backend="cvxpy"` |

No sampler, projection operation, general nonlinear optimizer, or automatic
calibration is advertised by this slice. `capabilities.linear_optimization`
identifies the domains with `maximize_linear`. Minimization can be computed by
negating coefficients and reversing/negating the returned bounds.

`SimplexSet` requires `basis="mass"` with `unit="mass_fraction"`, or
`basis="mole"` with `unit="mole_fraction"`. Its sum constraint applies to
physical fractions, not their normalized deviations. It does not enforce the
bounds from parameters used to build its space; use an intersection, as above.
Its known feasible point is a pure-component vertex, so a nominal mixture need
not be its nonemptiness witness.

`BudgetSet` uses the envelope's nominals and scales, including asymmetric
physical bounds. The budget is nonnegative and can be fractional. A budget
larger than the coordinate count is allowed and redundant. It is neither a
probability nor a count of discrete outages.

`EllipsoidSet.shape` is a full, explicitly name-keyed symmetric positive-definite
matrix in normalized coordinates. The center is the space's nominal point.
The radius is nonnegative; zero yields the center only. Singular matrices and
estimated condition numbers above `1e12` are rejected. The shape is a declared
geometric restriction; covariance calibration and confidence-level inference
remain the caller's responsibility. Coordinate bounds are not implied.

Membership returns `inside`, `outside`, or `unknown`, with domain identity,
realization and residual evidence. Malformed coordinates remain input errors.
Box and finite membership use exact declared comparisons. For the new numerical
geometries, a nonpositive residual is inside, a residual above the classification
band is outside, and a small positive residual remains unknown. Equality uses
absolute residuals. Ellipsoid membership uses the equivalent Cholesky norm
residual `norm(L^-1 z) - radius`. The default band is `1e-10`; supported values
range from `1e-10` to `1e-4`. An envelope violation remains outside even if tiny.
Unrepresentable arithmetic returns unknown. `contains` raises when membership
is unknown, so it cannot silently turn an unresolved boundary into a decision.

These are floating-point checks against the declared mathematical geometry.
The band does not enlarge the set; neither these checks nor their support
bounds are interval-arithmetic or exact symbolic proofs.

## Linear support and coupled restrictions

`LinearConstraint` and `maximize_linear` use dimensionless coefficients on
normalized coordinates. Missing named coefficients mean zero; undeclared names
are rejected. A returned point uses physical coordinate values. If an application
starts with a physical linear expression, it must explicitly transform its
coefficients and constant term into this convention.

```python
from operange import (
    BoxSet, BudgetSet, EllipsoidSet, LinearConstraint, Parameter, PolytopeSet,
)

loads = BoxSet(tuple(
    Parameter(name, "fraction", 0, 0, 0.25, 0.25, "Synthetic load envelope")
    for name in ("source_derating", "demand_increase")
))
source, demand = loads.space.names
coupled = PolytopeSet(loads, (
    LinearConstraint("joint_excursion", {source: 1, demand: 1}, rhs=1),
))
support = coupled.maximize_linear({source: 2, demand: 1})
assert support.status == "optimal"
assert abs(support.upper - 2) <= support.tolerance
assert coupled.membership(support.point).status == "inside"

budget = BudgetSet(loads, budget=1.5)
budget_support = budget.maximize_linear({source: 2, demand: 1})
assert abs(budget_support.upper - 2.5) <= budget_support.tolerance

correlated = EllipsoidSet(loads.space, shape={
    source: {source: 1, demand: 0.5},
    demand: {source: 0.5, demand: 1},
})
ellipsoid_support = correlated.maximize_linear({source: 1, demand: 1})
assert ellipsoid_support.status == "optimal"
```

The ellipsoid in that example includes negative normalized excursions and
extends beyond the box. Intersect it with `loads` when those physical bounds
are also required. `Intersection((loads, correlated), backend="cvxpy")` enables
checked support with the optional extra; omitting the backend retains
membership-only behavior.

`PolytopeSet` accepts `relation="le"` or `"eq"` per constraint. Its envelope is
mandatory and finite. A verified `feasible_point` in physical coordinates is
required; the envelope nominal is tried when it is omitted. Construction never
invokes an optimizer to guess nonemptiness. An off-nominal equality domain can
be declared by providing an explicit feasible point. If no supplied point can
be verified, construction fails rather than producing an empty audit domain.

`LinearSupport` is a domain-operation evidence record, separate from the three
process result families. It contains:

- Domain reference and normalized objective coefficients.
- A feasible physical point and its attained lower bound.
- An analytical or dual upper bound, numerical tolerance and evidence.
- `status="optimal"` for a gap no greater than tolerance, `"bounded"` for an
  open gap, or `"unresolved"`/`"unsupported"` without fabricated bounds.

Simplex support chooses a maximizing vertex. Budget support allocates its
deviation budget in descending absolute-weight order, respecting directional
box bounds. Ellipsoid support uses `radius * sqrt(w.T @ shape @ w)`, evaluated
through a Cholesky factor. If rounding puts an analytical candidate just outside
the set, it is moved inward toward a known feasible point; the lower bound
uses that actual candidate and the upper bound retains the analytical value.

For a polytope `G z <= h`, the backend solves primal and dual LPs through the
existing checked SciPy/HiGHS wrapper. Primal candidates must independently pass
domain membership. Nonnegative dual multipliers `lambda` give the upper bound

```text
h.T @ lambda + support_of_envelope(w - G.T @ lambda)
```

The envelope term accounts for a nonzero dual equality residual; a dual solve's
termination label alone is insufficient. Evidence retains the multipliers,
residual correction, rows and solver outcomes. On solver failure, the known
feasible point and finite box still supply valid, potentially loose bounds.
False infeasible/unbounded termination cannot override that nonempty bounded
domain. A closed gap can establish an optimum even when a backend fails,
provided independent bounds suffice.

The formulation follows standard support functions and weak duality as set out
in [Boyd and Vandenberghe, *Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
LP backend behavior is documented in
[SciPy's HiGHS interface](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html).

## Composition and portable declarations

`Intersection` lifts each factor's restrictions onto named coordinates in a
joint space. Factors may cover subsets. `Product` requires disjoint coordinate
names and permits every combination. `Union` requires matching coordinate names
and admits any factor without replacing the union by its convex hull.

Shared coordinates must have compatible names, units, vector axes and labels.
If their nominal, scale or provenance definitions differ, supply an explicit
joint `space`; this chooses joint metadata without rewriting a factor's own
normalization. Every composition requires a verified feasible point. It tries
the joint nominal and suitable factor witnesses, or checks a caller-supplied
`feasible_point`. Failure to find one is an input error, not proof of emptiness.

Intersection/product membership is outside if any factor is outside and inside
only if all are inside. Union membership is inside if any factor is inside and
outside only if all are outside. Other combinations remain unknown. Evidence
retains the constituent membership checks. Composition does not imply
probabilistic independence. Products sum support bounds over disjoint factors;
unions take the maximum across all branches and keep an actual branch witness.
These operations require every factor to provide normalized linear support.
Convex intersections can opt into the checked CVXPY backend. See the
[composition guide](../docs/uncertainty-compositions.md) for coordinate transforms,
evidence, nested support and worked engineering examples.

```python
import json
from operange import domain_from_json

encoded = json.dumps(domain.to_manifest(), allow_nan=False)
restored = domain_from_json(encoded)
assert restored.ref == domain.ref
assert restored.membership(point).status == "inside"
```

`domain_from_manifest` and `domain_from_json` reconstruct only the explicitly
supported built-in kinds. They validate fields, axes, numeric values and
nonemptiness witnesses without executing a model or LP. JSON rejects duplicate
keys and nonfinite constants. Manifests retain provenance; content digests
identify declarations rather than authenticating their author. Custom domain
protocols can be used at runtime but have no automatic deserializer.

## Process claim scope

These geometries are consumable through `UncertaintySet`, but domain support
functions do not establish recoverability of a nonlinear process or causal
controller. The existing heat adapter still supports its declared scalar box
binding, and storage retains its finite two-period binding. Substituting one of
these geometries returns a structured unsupported result where the adapter has
no applicable method. No solver executes and no universal claim is inferred.

The [fixed affine adapter](affine-process-adapter.md) now uses these support
bounds for named engineering requirements, including unresolved and unsupported
paths. General trajectories, fault semantics, model ensembles,
implicit domains and probability-law ambiguity remain separate later work.
