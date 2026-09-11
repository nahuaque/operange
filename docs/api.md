# Public API and capability scope

The alpha package has two supported import surfaces: `operange`
for shared engineering concepts and `operange.reference` for the
bounded reference models. The API is still being developed, with consumer
examples maintained alongside it.

## Shared declarations

Import these from `operange`:

| Purpose | Public names |
| --- | --- |
| Claim and adapter contract | `Claim`, `ModelAdapter`, `AdapterCapabilities`, `Capability`, `bind_contract`, `rejected_result` |
| Named inputs and coordinates | `Parameter`, `VectorParameter`, `Coordinate`, `ParameterSpace`, `Axis` |
| Domain protocol and evidence | `UncertaintySet`, `DomainCapabilities`, `MembershipCheck`, `LinearSupport` |
| Concrete domains | `BoxSet`, `FiniteSet`, `Scenario`, `SimplexSet`, `BudgetSet`, `EllipsoidSet`, `PolytopeSet`, `LinearConstraint` |
| Domain composition and loading | `Intersection`, `Union`, `Product`, `domain_from_manifest`, `domain_from_json` |
| Operating permissions and severity | `DecisionRule`, `RecoursePolicy`, `Distance`, `NormalizedLInf` |
| Fixed affine model | `AffineProcessAdapter`, `AffineOutput`, `AffineTerm`, `AffineRequirement` |
| Adjustable linear model | `LinearProcessAdapter`, `LinearControl`; reuses `AffineOutput`, `AffineTerm`, `AffineRequirement` |
| Supplied time profile | `PiecewiseLinearProfile` |

`Claim(adapter, domain, recourse, requirements=None, distance=None)` binds a
model's declared engineering requirements to an aligned uncertainty space and
operating contract. Omitted requirements select all declared requirements.
Its `contract` and `capabilities` describe that binding. Queries are
`evaluate_result`, `sensitivity_result`, `audit_result`, `boundary_result` and
`breaking_result`; their options depend on the adapter and query.

Domains declare their own membership and support capabilities independently
of the adapter. Composition does not automatically add a numerical optimizer.
Normalization scales are explicit; vector coordinates preserve their labels.
Units describe quantities and derivatives without converting values.

## What the built-in adapters establish

| Adapter | Evaluation and sensitivity | Robustness |
| --- | --- | --- |
| `AffineProcessAdapter` | Fixed affine responses; analytical first derivatives in physical or normalized coordinates | Direct physical enumeration of finite sets; linear-support audits over supported boxes, simplexes, budgets, ellipsoids and polytopes; no adjustable recourse or distance searches |
| `LinearProcessAdapter` | Joint feasibility of bounded controls, coupled operating limits and selected requirements at a verified member; no dispatch derivatives | Complete `FiniteSet` audits under fixed or fully observed static operation; exact physical feasibility checks and bounded-control infeasibility certificates; no continuous-domain audits or distance searches |
| `reference.HeatRecoveryAdapter` | Static constant-COP heat model with declared recourse; analytical local and directional first derivatives where supported | Box audits, boundary and positive-shortfall breaking searches within its verified static model and normalized distance |
| `reference.ThermalStorageAdapter` | Two-period finite-tree dispatch with fixed, causal or perfect-foresight permissions; no sensitivity operator | Audits of the declared finite tree, including incompatible futures; no continuous-domain radius or general multistage search |
| `reference.StartupLoadAdapter` | Supplied piecewise-linear load profiles with fixed start times, amplitude/duration scales and timing jitter; peaks and time integrals | Complete finite uncertainty enumeration with continuous-time coverage of the declared interpolation; no adaptive scheduling, derivatives or distance searches |
| `reference.HeatCascadeAdapter` | Steady-state sensible-heat cascade at fixed ΔTmin; minimum utilities, maximum heat recovery and all pinch locations | Finite-scenario utility-target screening only; no installed-network feasibility, derivatives or distance searches |
| Consumer `ModelAdapter` | Only the operations its implementation supports | Only the guarantees its evidence establishes |

Inspect `claim.capabilities` for the exact binding and result diagnostics for
the requested operator. A domain's membership support is not a universal audit
guarantee. Affine derivatives are ambient coordinate derivatives, not an
implicit choice of tangent coordinates on a constrained composition surface.
Heat derivatives retain their operating-mode and locality qualifications.

Finite affine audits do not require nominal values or normalization scales.
They evaluate every declared physical realization and report `complete_finite`
coverage, retaining duplicate scenario labels. Affine residuals use exact rational
arithmetic on the declared floats and round toward positive infinity for checks.
Continuous affine audits carry coefficient-rounding corrections and directed
support bounds into physical residual units. Unsupported or numerically unresolved
calculations retain their diagnostics.

`LinearControl(name, unit, lower, upper, physical_kind="declared_process_control")`
declares finite physical bounds. `LinearProcessAdapter(name, input_space, outputs,
requirements, controls, operating_limits=(), solver_tolerance=1e-9)` reuses affine
expressions in physical inputs and controls. All operating limits and control
bounds always apply; only `requirements` can be selected by a claim.
`model.as_claim(domain)` defaults to independent static adjustment after observing
every input. Explicit rules must use stage `"operation"`; a rule can instead fix
its control. A `"fixed"` policy must fix every control. Partial observation and
causal or perfect-foresight modes are unsupported in this adapter.

Linear evaluation includes the selected requirements in its feasibility problem.
An infeasible response proves their joint incompatibility with the model and
operating permissions; it does not assign individual requirement violations to
a nonexistent dispatch. A feasible result returns checked controls and residuals,
without claiming a unique or optimal dispatch. See the
[linear dispatch guide](linear-dispatch.md) for the example and certificate.

Heat-recovery results retain a feasible objective lower bound and a conservative
analytical upper bound. If the feasible response misses the service requirement
but the upper bound permits recovery, the audit is inconclusive. Heat derivative
overflow or underflow returns an unresolved result without a fabricated derivative.
The reference `OperationResult` also exposes `capacity_lower_mw` and
`capacity_upper_mw`; its attained response remains in `maximum_heat_mw`.

The `ScenarioTree` reference type is specifically a two-period storage tree.
General trajectories, degradation, faults, implicit sets, model ensembles,
probability-law ambiguity and general multistage policies are future work.
The [startup slice](startup.md) adds bounded event-profile semantics without
claiming a general `TrajectorySet` or dynamic simulator.

## Results and portable contracts

| Purpose | Public names |
| --- | --- |
| Result families and loading | `EvaluationResult`, `SensitivityResult`, `RobustnessResult`, `result_from_json` |
| Contract registry | `ProcessContract`, `QuantitySpec`, `QuantityValue`, `ConstraintSpec`, `ConstraintCheck`, `ArtifactRef` |
| Evidence records | `Evidence`, `Measurement`, `Diagnostic`, `Membership` |
| Query payloads | `EvaluationPayload`, `SensitivityPayload`, `RobustnessPayload`, `Derivative` |
| Robustness evidence | `Bound`, `Objective`, `Coverage`, `Witness`, `Search`, `Cardinality` |

Each result exposes `to_dict()` and `to_json()` with schema
`process_result/v1`. Execution status is distinct from physical feasibility
and the claim verdict. Missing or unsupported derivatives are not zero values;
solver failure alone is not a certified breaking witness. Inspect coverage,
bounds and witness evidence before interpreting a verdict as global.

`to_dict(compact=True)` and `to_json(compact=True)` opt into
`process_result_bundle/v1`, containing `result`, `contracts` and `evaluations`.
The artifact tables are keyed by existing content identities; each result's
`artifacts` names its contract and supporting evaluations. Expanding the tables
reconstructs the original v1 result and its identity. `result_from_json` and the
typed `from_json` methods accept both formats. Older readers need the default
v1 export. Neither the underlying result IDs nor the default schema changes.

The compact reader rejects altered content, missing or unused artifacts, duplicate
references and nested evaluation cycles. The result validators also require a
quantity repeated in a constraint residual or attained objective to match its
stored value. Loading validates these records and identities without executing
a model or re-proving the recorded evidence.

Results serialize model and operating declarations, not Python adapter code.
Content identities support evidence joins and integrity checks. They do not
replace independent verification of a consumer model or solver.

## Reference-model imports

Import the following from `operange.reference`:

- Heat model: `HeatRecoveryClaim`, `HeatRecoveryAdapter`, `HeatDemand`,
  `HeatRecoveryDesign`, `RepairAction`, `FrozenHeatRecoveryContract`,
  `example_claim`.
- Heat numerical records: `ClaimAudit`, `OperationResult`, `RepairComparison`,
  `ThresholdResult`.
- Storage model: `ThermalStorageClaim`, `ThermalStorageAdapter`, `ScenarioTree`,
  `StorageDesign`, `StorageRepairAction`, `StorageRequirement`, `StorageScenario`,
  `example_storage_claim`.
- Storage numerical records: `StorageAudit`, `StorageDispatch`,
  `StorageRepairComparison`, `StorageWitness`.
- Supplied startup profiles: `StartupEvent`, `StartupLoadAdapter`; these use
  the common evaluation and robustness result families directly.
- Heat-integration targets: `SensibleHeatStream`, `HeatCascadeAdapter`; these
  also use the common result families. See the [heat-cascade guide](pinch.md)
  for target versus network feasibility and exact finite-scenario coverage.

These are model-specific engineering APIs. Repair comparisons audit caller
changes in caller order; they do not attach costs or choose a preferred repair.

## Implementing an adapter

A `ModelAdapter` exposes `base_contract`, `input_space`, `describe(claim)`,
`capabilities(claim)` and `run(claim, operation, realization, options)`.
`describe` must bind the actual domain, operating permissions and requirements;
`bind_contract` is a public helper for that operation. `run` returns the matching
typed result family for that exact contract. `rejected_result` constructs a
typed diagnostic result when a query cannot be executed.

The shared `Claim` checks contract alignment and result family/identity. The
adapter remains responsible for physical evaluation, numerical validation,
coverage and any proof obligations. Implement only supported capabilities and
retain unresolved outcomes when the numerical evidence cannot decide a claim.

## Experimental namespace

`operange.experimental` contains opt-in quadratic/backend research.
It is excluded from the supported API promise and has no default dependency on
residopt or CVXPY. That investigation is parked. Its presence does not extend
the capabilities of the shared or reference adapters.
