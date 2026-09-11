# Shared process DSL

Implemented on the experimental branch, 2026-09-11. The DSL binds uncertainty,
requirements, operating permissions and severity to a process model adapter.
Heat recovery and thermal storage are reference models for this interface.
The package's product boundary remains the engineering DSL and its structured
results; financial valuation, ranking and presentation belong downstream.

## Implemented interfaces

| Interface or primitive | Behavior in this slice |
| --- | --- |
| `Coordinate` | Named scalar with units and provenance; optional nominal, positive scale and vector-component metadata |
| `ParameterSpace` | Validates named realizations, checks coordinate/unit/axis alignment, flattens labeled vectors, and normalizes coordinates |
| `Parameter` | Existing bounded scalar; exposes its shared `Coordinate` |
| `VectorParameter` | Bounded one-axis vector with label-keyed values, units, scales and provenance |
| `UncertaintySet` | Protocol for coordinate space, manifest, identity, capabilities and membership |
| `BoxSet` | Implements that protocol; retains its existing bounds and convenience methods |
| `Scenario`, `FiniteSet` | Generic finite physical realizations with stable scenario labels, membership and enumeration |
| `PolytopeSet`, `SimplexSet`, `BudgetSet`, `EllipsoidSet` | Continuous geometries with membership and linear support bounds |
| `Intersection`, `Union`, `Product` | Typed composition with named alignment, a nonemptiness witness and three-valued membership |
| `LinearSupport` | Dimensionless domain extrema with a feasible point, lower/upper bounds and evidence |
| `Distance`, `NormalizedLInf` | First-class distance interface and the existing scaled maximum-deviation metric |
| `DecisionRule`, `RecoursePolicy` | Control identities, units, stages, permitted observations and optional fixed decisions |
| `Claim` | Binds a model adapter, domain, recourse, named engineering requirements and optional search distance |
| `ModelAdapter`, `AdapterCapabilities` | Declare supported queries and their scope, then return the shared result types |
| `AffineProcessAdapter`, `AffineOutput`, `AffineTerm`, `AffineRequirement` | Fixed affine physical responses, analytical derivatives and requirement audits using domain support bounds |

`ParameterSpace` contains named scalar coordinates, including components of a
labeled `VectorParameter`. General tensors and automatic unit conversion remain
future work. Changing a unit label is never treated as converting a value.
See [uncertainty geometries](uncertainty-geometries.md) for constructors,
normalization conventions, composition and portable domain manifests.

## One interface for both reference models

```python
from operange import Claim, NormalizedLInf
from operange.reference import (
    HeatRecoveryAdapter, example_claim, example_storage_claim,
)

heat_model = example_claim()
heat = Claim(
    adapter=HeatRecoveryAdapter(heat_model),
    domain=heat_model.uncertainty,
    recourse=heat_model.recourse_policy,
    requirements=("heat_demand",),
    distance=NormalizedLInf(heat_model.uncertainty.space),
)
# Equivalent convenience binding: heat_model.as_claim()

point = {"source_derating": 0.1, "demand_increase": 0.05}
evaluation = heat.evaluate_result(point)
derivative = heat.sensitivity_result(point)
breaking = heat.breaking_result(shortfall_margin_mw=0.01)

storage_model = example_storage_claim()
causal = storage_model.as_claim()
fixed = storage_model.as_claim("fixed", fixed_preparation_power_mw=0.5)
foresight = storage_model.as_claim("perfect_foresight")

for claim in (heat, causal, fixed, foresight):
    print(claim.capabilities.audit.scope)
    print(claim.audit_result().payload.verdict)
```

The existing model-specific structured methods now dispatch through `Claim`.
The original low-level numerical methods remain available for compatibility.
The heat boundary remains `4/7` and its 0.01 MW shortfall witness remains
`201/350`. The storage example retains fixed/causal failure and perfect-foresight
success, including the minimal two-future causal conflict.

`Claim.requirements` references constraint identities declared by the adapter.
It does not parse or execute arbitrary expression strings. Each reference must
identify an engineering requirement; control permissions must match the model's
control roles and units. An adapter may reject a valid declaration that its
algorithm cannot evaluate, including a changed subset of requirements.

## Domain operations and proof capabilities

Domain capabilities and adapter capabilities are separate:

- `BoxSet` provides membership and separable linear support. It does not
  advertise a sampler or enumeration.
- `FiniteSet` provides membership and exact enumeration. It assigns no
  probabilities and does not infer a nominal point or normalization scale.
  With explicit normalization it also provides enumerated linear support.
- Polytope, simplex, budget and ellipsoid domains provide membership and
  `maximize_linear` on normalized coordinates. Composition provides membership
  only. These operations do not confer process certification capabilities.
- The [affine adapter](affine-process-adapter.md) uses domain support bounds
  for audits of declared affine requirements under fixed operation. Its
  sensitivities explicitly differentiate the ambient affine map.
- The heat adapter supports its applicable box domains and fully observed
  static recourse. Its full-domain audit relies on this model's monotonicity.
- The storage adapter supports the finite futures bound to its two-period
  model, with its declared preparation/event permissions. It solves the
  futures jointly. A point evaluation would change that scope and is rejected.

`MembershipCheck` distinguishes `inside`, `outside` and `unknown`, carries the
domain identity and evidence, and rejects malformed coordinates as input errors.
Box/finite comparisons use exact declared values and inclusive box bounds.
The new numerical geometries retain unknown status for small positive residuals
within their classification band. They do not widen the uncertainty domain by
a solver tolerance. Finite membership reports all matching scenario identities
when multiple labeled events have the same numeric realization.

An unsupported combination returns `execution="unsupported"` and no semantic
claim conclusion or solver execution. For example, substituting a `FiniteSet`
for the heat adapter's `BoxSet` does not transfer its box proof to that domain.
The adapter's declared capability is eligibility for a calculation; the returned
evidence still determines whether any particular solve actually establishes it.

Query options can narrow that eligibility further. Heat sensitivity still
supports analytical optimized-response first derivatives only. The storage
adapter does not yet provide sensitivity or continuous-distance searches.

## Information and severity

The storage tree exposes physical futures through `tree.domain`. Preparation
observations are kept in the recourse/model binding. Changing information timing
therefore changes the operating contract while retaining the physical domain.

`DecisionRule.information(observations)` selects only the permitted observation
fields and rejects missing required fields. Storage uses this selection to form
shared preparation decisions. Its causal preparation rule cannot inspect event
identity; perfect foresight explicitly permits that observation. The adapter
validates stage timing and supported policy classes. A rule that labels access
to future event identity as causal is rejected by this adapter.

`RecoursePolicy` declares permissions and fixed decisions. An empty fixed policy
represents a model with no adjustable controls. It is not yet an
executable frozen controller, affine feedback policy or arbitrary policy program.

`NormalizedLInf` measures deviations in a named coordinate space, including
out-of-domain points, independently of membership. Its declared nominal point
is the default baseline; `measure(..., baseline=...)` can compare two explicit
points. This geometric measure has no probability interpretation.

Changing a search distance does not change process physics. The current heat
search backend supports only the box's declared nominal and scales. A different
distance remains usable for direct measurement, while the corresponding search
returns unsupported. There is no silent substitution of a supported metric.

## Consumer adapters and portable results

A consumer can implement `ModelAdapter` without subclassing either reference
model. It provides a base `ProcessContract`, an input `ParameterSpace`, a
description bound to the requested claim, query-specific capabilities, and a
`run` method returning `EvaluationResult`, `SensitivityResult` or
`RobustnessResult`. The framework checks the returned family and contract
identity. The adapter remains responsible for validating its model applicability,
observation timing and numerical proof obligations.

Result exports retain `process_result/v1`. Domain manifests and the explicit
recourse/requirement declarations live inside the bundled process contract;
runtime adapters and Python callables are not serialized. New declarations
change content fingerprints. Previously exported v1 results remain readable,
and the separate frozen static heat-contract format is unchanged. Reading a
result does not instantiate its adapter or execute its model.

## Next slices

The [geometry slice](uncertainty-geometries.md) implements vector parameters,
`PolytopeSet`, `SimplexSet`, `BudgetSet`, `EllipsoidSet`, and typed composition.
The [fixed affine adapter](affine-process-adapter.md) now exercises their support
bounds in complete engineering claims and portable results. The heat and storage
reference adapters retain their current proof scopes. Adjustable affine recourse
and distance searches remain unsupported.

General trajectories, multistage scenario trees, degradation, faults, implicit
membership and model ensembles follow concrete engineering cases. Probability-law
ambiguity remains a separate later track. Monetary repair helpers and Markdown
renderers now live in consumer examples; see the
[getting-started guide](../docs/index.md).
