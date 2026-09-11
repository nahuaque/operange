# Uncertainty and recourse

Part of the [process exploration](README.md). The [shared DSL slice](dsl-primitives.md)
implements scalar/vector coordinates, continuous and finite domains, typed
composition, explicit recourse permissions, normalized distance and generic
claims. The [geometry slice](uncertainty-geometries.md) documents supported
operations. The table distinguishes implementation from remaining design work.
These primitives belong to the [engineering DSL](package-boundary.md);
valuation, business selection, and presentation belong to downstream code.

## Separate inputs, physical states, and decisions

A process claim needs at least these distinct objects:

- `d`: design fixed before uncertainty is revealed, such as installed capacity.
- `u`: exogenous realization, potentially including a trajectory and fault mode.
- `m`: a credible model choice with a declared applicability region.
- `x`: process states constrained by the model, such as temperatures and holdup.
- `z`: operating decisions with bounds and a specified information pattern.

For example, weather and source availability can be uncertain inputs; heat-pump
COP can be a model output depending on temperatures and operating point.
Independently perturbing both the inputs and that output can create inconsistent
states. A residual COP uncertainty is possible, but it needs an explicit model
discrepancy definition rather than duplicating the same uncertainty.

The external admissibility set describes which disturbances are in scope.
The process model describes whether the design can operate under them. Do not
define admissibility using "the simulator converges" or "this design meets its
operating limits": either would remove potential failures before the audit.
An impossible external feed state may be excluded; an admissible feed that the
plant cannot handle must remain in the stress set.

## Candidate primitives

| Primitive | Meaning and required contract | Initial scope |
| --- | --- | --- |
| `Parameter`, `VectorParameter` | Named scalar or vector; units, nominal value, axes, bounds, source | Scalar and labeled one-axis vectors implemented |
| `BoxSet` | Coordinate bounds; no joint probability or probabilistic independence implied | Implemented through `UncertaintySet` |
| `FiniteSet` | Explicit named physical realizations, membership and enumeration | Implemented; no inferred probabilities |
| `PolytopeSet` | Coupled normalized linear inequalities and equalities within a finite box | Membership and checked support bounds implemented |
| `EllipsoidSet` | Nominal center, normalized positive-definite shape and radius | Membership and analytical support implemented; singular shapes rejected |
| `BudgetSet` | Normalized deviations and a simultaneous-deviation budget | Membership and analytical support implemented |
| `SimplexSet` | Nonnegative fractions summing to one; explicit mass or mole basis | Membership and analytical support implemented |
| `ImplicitSet` | Membership oracle with residuals, tolerances, and unknown status | Simulator slice |
| `TrajectorySet` | Paths on an explicit grid, with bands, ramps, and event constraints | First dynamic slice |
| `DegradationSet` | Path evolution, age, rate bounds, and reset semantics | Later |
| `ScenarioTree` | Branching observations with shared histories and decision stages | Two-period storage tree implemented; general multistage trees later |
| `FaultSet` | Discrete modes, excluded combinations, and concurrency/event budgets | Finite modes first |
| `ModelEnsemble` | Versioned model alternatives and applicability; no implicit weights | Later |
| `AmbiguitySet` | Probability laws over a specified realization space | Deferred |
| `Intersection`, `Union`, `Product` | Typed composition with explicit coordinate alignment | Membership and verified nonemptiness witnesses implemented |
| `RecoursePolicy` | Frozen controller or admissible class of decision rules | Explicit permissions implemented; executable frozen controllers later |
| `Distance` | Declared severity objective, normalization, and baseline | First-class normalized L-infinity implemented |
| `Claim`, `RepairAction` | Requirement and candidate change to the design or operating contract | Generic claim binding implemented; repair helpers remain model-specific |

Units apply at the boundary and in reports. Internal numerical coordinates may
be scaled, but conversions must be recorded. Temperature offsets require
special care: an absolute Celsius temperature and a Celsius difference do not
have the same conversion rule. Time grids, composition bases, and vector labels
are part of the type contract, not positional conventions left to the caller.

## Composition is a typed operation

An intersection combines restrictions on a common joint realization. A
convenience implementation can lift component restrictions onto explicitly
named coordinates, then intersect them. It must reject conflicting units,
axes, or definitions instead of silently joining arrays by position.

A product permits every combination of values in its factors; it makes no
probabilistic independence assertion. A union admits alternatives and can
destroy convexity. Discrete alternatives should retain their mode labels in
witnesses.

Conceptual construction:

```python
# Design sketch, not an existing API.
external_conditions = product(weather_paths, feed_compositions, outage_modes)
uncertainty = intersection(
    external_conditions,
    joint_weather_feed_restrictions,
    external_state_feasibility,
)
```

Dependencies belong either in this joint input set or in the process model,
according to their meaning. A common outdoor temperature affecting both load
and cooling conditions should be represented once and propagated through the
model. A correlation estimated from history is an empirical restriction with
provenance; it is not automatically a physical law.

Every set should expose a schema, provenance, and a membership result:
`inside`, `outside`, or `unknown`. Sampling, projection, bounds, support
functions, and symbolic compilation are optional capabilities. A sampler must
identify its sampling scheme; returning members does not imply uniform or
representative sampling. A membership predicate alone cannot generally supply
global optimization or even efficient sampling.

Check nonemptiness and nominal membership explicitly where possible. An empty
set is an invalid audit domain, rather than a useful vacuous pass. If emptiness
cannot be resolved, report that limitation. A deliberately off-nominal event
set can exclude the nominal point, but must say so; a distance to that set is
not automatically a radius around a supported baseline.

## Recourse determines the claim

Let `F(d, u, m)` contain state/control pairs satisfying the model equations and
the declared hard operating constraints. Let `g_j <= 0` express the claim's
requirements. Three different contracts are useful:

1. **Fixed operation:** prescribed controls must work for every realization.
2. **Static adjustable operation:** after observing all relevant uncertainty,
   an admissible control/state pair must exist for each realization.
3. **Causal operation:** one allowed policy must work across the entire set,
   using only observations available at each decision time.

Static adjustable feasibility is

$$
\forall (u,m)\in\mathcal U,\quad
\exists(x,z)\in F(d,u,m):\quad g_j(d,x,z,u,m)\le0\quad\forall j.
$$

That quantifier order permits controls to depend on the model choice. Use it
only if that choice is observable before the decision. If model identity is
unknown to the controller, decisions for indistinguishable observations must
be shared across the model alternatives too.

For causal operation, with observation history `h_t`, the contract becomes

$$
\exists\pi\in\Pi_{\mathrm{causal}}\quad
\forall(u,m)\in\mathcal U:\quad z_t=\pi_t(h_t),
\quad\text{model equations and claim requirements hold}.
$$

Record whether the current disturbance is observed before or after the current
action. Shared scenario-tree histories imply shared actions. Solving a separate
full-horizon dispatch for each realized path gives a perfect-foresight
benchmark; it does not establish the causal claim.

For a **frozen** controller, evaluate that controller directly. Its failure
establishes failure of that policy, but may leave another policy feasible.
Likewise, failure of an affine policy class does not prove failure of every
possible controller. Hardware feasibility, operational flexibility, and safe
shutdown should have separately named claims and control permissions.

## Trajectories and faults need timing

A path contract includes timestamps, step duration, initial state, terminal
conditions, observation delay, and interpolation rules. Example restrictions
include ramp rates `abs(T[t+1] - T[t]) <= r * dt`, energy budgets weighted by
`dt`, and maximum heatwave duration. A sum of squared pointwise shocks must
state whether it approximates a time integral and how its units are scaled.

Storage, minimum run times, and equipment recovery connect periods through
model state. Monotone degradation between cleanings can be part of the path
model; a cleaning schedule chosen by the operator is a decision with physical
downtime and observation rules. Consumers can value its cost separately.
It must not become an adversarial reset by accident.

A fault budget must distinguish "one simultaneous outage" from "one outage
event over the horizon". Include duration, detection delay, recovery, and common
cause exclusions. A stuck sensor changes the observation model; it does not
merely change the true process parameter seen by an omniscient controller.

## Realization uncertainty and distributional ambiguity

A robust claim quantifies over `u in U`. A distributionally robust claim
quantifies over laws `P in P_set`, for example

$$
\sup_{P\in\mathcal P}\mathbb E_P[L(d,\pi,u)]\le b
\quad\text{or}\quad
\sup_{P\in\mathcal P}P(\text{claim violation})\le\alpha.
$$

These require a loss or event, a measurable sample space, and a declared policy.
A single unfavorable realization does not refute either statement; evidence
must establish the corresponding worst-case expectation or probability bound.
Never intersect a set of probability laws with a set of physical vectors as if
they had the same type.

Historical calibration should retain training periods, preprocessing, support,
dependence assumptions, and held-out evaluation. Geometric distance and a
radius fitted to history do not by themselves give a failure probability.
Distributional ambiguity is deferred until the deterministic contracts work.
