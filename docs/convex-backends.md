# Optional CVXPY backends

Install the existing optional extra to enable prepared dispatch programs and
checked support for convex intersections:

```bash
python -m pip install 'operange[cvxpy]'
# From a checkout:
uv run --extra cvxpy python -m examples.convex_backends
```

## Reuse dispatch and relief programs

Select the backend on a linear dispatch query:

```python
audit = claim.audit_result(backend="cvxpy")
diagnosis = claim.evaluate_result(
    {"dryer": 12, "evaporator": 8},
    backend="cvxpy",
    diagnose=True,
    relief={"constraint": "shared_fuel", "maximum": 2},
)
```

The backend compiles the normalized constraint matrix once and supplies new
right-hand sides and objective coefficients through CVXPY parameters. A bounded
cache holds up to 16 prepared programs per thread. It is separate from immutable
model declarations and portable results. Different row matrices or program types
compile separately; a repeated compatible scenario reuses the prepared problem.

Dispatch uses a phase-I linear program. Its primal candidate proposes controls,
and its row duals can propose a contradiction when no feasible controls are
found. Relief uses a linear objective and row duals to propose a lower bound.
Both paths retain the existing exact checks on physical controls, original
constraints, contradiction certificates and relief candidates. Solver termination
alone cannot establish feasibility, failure, or minimum relief.

The default remains `backend="scipy"`. The CVXPY LP backend currently also uses
SciPy/HiGHS: the change is the reusable modeling layer and access to primal and
dual candidates from one solve. It does not imply that every LP will run faster.
Backend selection appears in the query and solver evidence, while engineering
contract identities remain the same. Missing CVXPY returns `unsupported` with
installation guidance; invalid backend names return `invalid`.

## Optimize over a convex intersection

An intersection can explicitly enable linear support:

```python
joint = Intersection((envelope, ellipsoid), backend="cvxpy")
support = joint.maximize_linear({"dryer": 1, "evaporator": 1})
audit = model.as_claim(joint).audit_result()
```

Weights apply to **normalized coordinates**, and the returned point uses physical
units. The supported factors are `BoxSet`, `PolytopeSet`, `BudgetSet`,
`SimplexSet`, and `EllipsoidSet`, including those nested in intersections and
disjoint products. Every coordinate needs a nominal and scale. Different factor
normalizations require an explicit joint `space`; units and vector axes must
align. Unions, finite scenario sets and custom factor types are unsupported by
this backend. An intersection without a backend retains membership-only behavior.

In the worked example, the dryer envelope ends at 11 MW, while an ellipsoid
limits the joint normalized deviations. The maximum total demand is about
18.732 MW over their intersection. The box alone reaches 19 MW; the ellipsoid
alone reaches about 18.828 MW. A header rated at 18.75 MW therefore passes the
joint envelope and fails each broader factor considered separately.

```{literalinclude} ../examples/convex_backends.py
:language: python
:start-at: def intersection_example():
:end-before: def run_example():
```

Existing affine audits and continuous frozen affine-controller audits consume
this support operation automatically. Frozen controllers preserve the backend
declaration in their domain manifest. Loading an artifact does not import CVXPY;
executing the support query requires the extra. This does not add a continuous
audit for freely adjustable recourse.

## How intersection evidence is checked

CVXPY/Clarabel proposes a common point and a split of the objective across the
factors. Each factor already has a checked support routine. If local coordinates
satisfy `y_i = alpha_i * z + beta_i`, any finite split vectors `a_i` give:

```text
h_intersection(w) <= sum(h_factor_i(a_i) - dot(a_i, beta_i))
                     + h_enclosure(w - sum(alpha_i * a_i))
```

The backend computes the coordinate transforms, residual and enclosure correction
using exact rationals of the declared floating-point values, then rounds the
upper bound outward. Factor support records, transforms, residuals and the final
bound are exported. The proof does not depend on approximate conic dual feasibility.

The physical witness is independently checked against every factor, including
exact linear equalities, simplex balances and ellipsoid quadratic forms.
Boundary candidates can move toward the declared feasible point to recover
membership after rounding. If solver candidates fail, a verified feasible point
and the finite enclosing box can still provide a conservative `bounded` result.
If nonemptiness or required arithmetic cannot be checked, support is unresolved.
In particular, a rounded simplex sum of one may fail the exact balance check;
provide a feasible point whose declared float values sum exactly to one.

## Measure before changing the default

Run the small repeatable benchmark from the checkout:

```bash
uv run --extra cvxpy python -m benchmarks.cvxpy_backend --repeats 50
```

It reports cold and median warm times for complete checked dispatch queries,
alternating scenario loads, plus intersection support with changing objectives.
The first CVXPY timing includes its lazy import. Tiny LPs can be dominated by
CVXPY and evidence-construction overhead; repeated compilation reuse is an
architectural benefit, not a general speed guarantee. Benchmark the intended
study sizes before changing backend defaults.
