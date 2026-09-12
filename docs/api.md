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
| Concrete domains | `BoxSet`, `FiniteSet`, `ConvexHullSet`, `Scenario`, `SimplexSet`, `BudgetSet`, `EllipsoidSet`, `PolytopeSet`, `LinearConstraint` |
| Domain composition and loading | `Intersection`, `Union`, `Product`, `domain_from_manifest`, `domain_from_json` |
| Operating permissions and severity | `DecisionRule`, `RecoursePolicy`, `Distance`, `NormalizedLInf`, `NormalizedL2` |
| Fixed affine model | `AffineProcessAdapter`, `AffineOutput`, `AffineTerm`, `AffineRequirement` |
| Adjustable linear model | `LinearProcessAdapter`, `LinearControl`; reuses `AffineOutput`, `AffineTerm`, `AffineRequirement` |
| Dispatch objectives | `LinearObjective`, `ControlTrackingObjective`, `ControlTarget` |
| Executable linear controllers | `AffineControlRule`, `AffineController`, `FrozenController` |
| Engineering changes | `EngineeringChange`, `compare_changes`, `ChangeComparison`, `ChangeResult`, `ContractComparison`, `FieldChange`, `RequirementSummary` |
| Supplied time profile | `PiecewiseLinearProfile` |

`Claim(adapter, domain, recourse, requirements=None, distance=None)` binds a
model's declared engineering requirements to an aligned uncertainty space and
operating contract. Omitted requirements select all declared requirements.
Its `contract` and `capabilities` describe that binding. Queries are
`evaluate_result`, `sensitivity_result`, `audit_result`, `boundary_result` and
`breaking_result`; their options depend on the adapter and query.

`claim.with_controller(controller)` binds a supported explicit controller to a
linear or storage claim while preserving its existing permissions. `claim.freeze()` then
captures the complete executable binding as a `FrozenController`. See the
[frozen-controller guide](frozen-controllers.md) for portable replay and scope.

`claim.compare_changes(changes)` re-audits named candidate claims and returns a
`ChangeComparison` with the baseline audit, ordered candidate audits, structural
changes, commitment/transition labels and observed requirement margins.
`compare_changes(baseline, changes)` is the equivalent function. Inputs use
`EngineeringChange(name, claim, provenance)`. The comparison has its own
`engineering_change_comparison/v1` envelope, read with
`ChangeComparison.from_json`; each embedded audit keeps its existing result
format. See the [engineering-change guide](engineering-changes.md) for conservative
service-equivalence rules, reference-model examples and portable evidence scope.

Domains declare their own membership and support capabilities independently
of the adapter. `Product` sums factor support bounds and `Union` takes their
maximum when every factor supplies normalized linear support; see
[combined uncertainty domains](uncertainty-compositions.md).
`Intersection(..., backend="cvxpy")` opts into checked support for supported
convex factors; see [optional convex backends](convex-backends.md).
Normalization scales are explicit; vector coordinates preserve their labels.
Units describe quantities and derivatives without converting values.

Adjustable dispatch can optimize a declared affine output or track physical
command targets with a convex quadratic objective. Feasibility and optimality
have separate evidence; see [operating objectives](dispatch-objectives.md).

## What the built-in adapters establish

| Adapter | Evaluation and sensitivity | Robustness |
| --- | --- | --- |
| `AffineProcessAdapter` | Fixed affine responses; analytical first derivatives in physical or normalized coordinates | Finite and supported linear-support audits; L∞ box/polytope distances by default, and optional convex-domain L∞/Euclidean distances; no adjustable recourse |
| `LinearProcessAdapter` | Joint bounded-control feasibility; opt-in irreducible row conflicts and single or jointly weighted operating-limit relief via `evaluate_result(diagnose=True, relief=...)`; no dispatch derivatives | Complete `FiniteSet` audits, plus continuous `BoxSet`/`ConvexHullSet` audits by checked generators under fixed or fully observed static operation; exact physical feasibility and infeasibility certificates; no adjustable distance searches |
| Linear model with an `AffineController` | Direct affine command execution and physical checks, retaining commands and violations; no optimizer or sensitivity query | Finite and supported continuous-envelope audits with command-rounding bounds and `fixed_policy_failure` witnesses; default L∞ box/polytope distances and optional convex-domain L∞/Euclidean distances; no causal state |
| `reference.HeatRecoveryAdapter` | Static constant-COP heat model with declared recourse; analytical local and directional first derivatives where supported | Box audits, boundary and positive-shortfall breaking searches within its verified static model and normalized distance |
| `reference.ThermalStorageAdapter` | Two-period finite-tree dispatch with fixed, causal or perfect-foresight permissions; no sensitivity operator | Audits of the declared finite tree, including incompatible futures; no continuous-domain radius or general multistage search |
| Storage model with a `reference.StorageController` | Direct causal replay with preparation signals, carried energy and terminal checks; no redispatch | Complete declared two-period paths, or partial coverage for unresolved paths; failures concern the saved controller; no continuous historical backtest |
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

Affine distance searches require an explicit `NormalizedLInf(space)` or
`NormalizedL2(space)` metric. Euclidean distances and extended convex domains
require `backend="cvxpy"`; the default retains L∞ searches over boxes and polytopes.
See [convex distances and joint relief](distance-and-relief.md) for supported
domains and dimensionless combinations of physical limit changes.
`boundary_result(distance_tolerance=1e-8)` targets a signed requirement residual
of zero or greater. `breaking_result(violation_margins={name: amount, ...},
distance_tolerance=1e-8)` requires one physical-unit margin per selected requirement,
each strictly greater than that requirement's tolerance. Searches minimize distance
to the union of the selected targets, checking every branch. The named origins
and scales come from the distance object and can differ from domain normalization.

`payload.search` records lower/upper distance bounds, resolution and a candidate
evaluation when available. `severity.thresholds` lists all physical targets;
`severity.reported_threshold_requirement` identifies the representative `threshold`
field and, when a candidate exists, its governing branch. An unreachable target
does not imply a passing claim audit. See the [failure-distance guide](failure-distance.md)
for proof semantics and numerical limits.

`LinearControl(name, unit, lower, upper, physical_kind="declared_process_control")`
declares finite physical bounds. `LinearProcessAdapter(name, input_space, outputs,
requirements, controls, operating_limits=(), solver_tolerance=1e-9)` reuses affine
expressions in physical inputs and controls. All operating limits and control
bounds always apply; only `requirements` can be selected by a claim.
`model.as_claim(domain)` defaults to independent static adjustment after observing
every input. Explicit rules must use stage `"operation"`; a rule can instead fix
its control. A `"fixed"` policy must fix every control. Partial observation and
causal or perfect-foresight modes are unsupported for adjustable dispatch.

Linear evaluation includes the selected requirements in its feasibility problem.
An infeasible response proves their joint incompatibility with the model and
operating permissions; it does not assign individual requirement violations to
a nonexistent dispatch. A feasible result returns checked controls and residuals,
without claiming a unique or optimal dispatch. See the
[linear dispatch guide](linear-dispatch.md) for the example and certificate.

`claim.audit_result(max_vertices=256, backend="scipy")` covers continuous boxes
and explicit convex hulls by checking every generator, with `analytical_domain`
coverage. It establishes real-valued dispatch existence through convex
combinations of feasible generator commands, not an executable controller or a
domain-wide dispatch optimum. Enumeration above the limit is rejected before
solving. General polytopes, other composed geometries and partial-observation
adjustable policies remain unsupported. See [continuous dispatch](continuous-dispatch.md).

`model.as_claim(domain, controller=controller, recourse=None, requirements=None)`
instead binds an `AffineController`. Default permissions use the observations
named by each rule; explicit `recourse` permissions must permit all its inputs.
This binding supports partial observation for an explicit rule at the single
`operation` stage. An infeasible response retains the actual commands and their
violations and concerns only that controller. `FrozenController.from_json`
loads a frozen binding, and `frozen.as_claim(new_domain)` explicitly rebinds the
same model and rule for new-scenario evaluation. The `frozen_controller/v1`
artifact and nested `affine_controller/v1` declaration each have content-derived
identities; individual evaluations and audits retain `process_result/v1`.

`ThermalStorageClaim.as_claim(controller=storage_controller)` binds a causal
`StorageController` with signal-indexed `StoragePreparationRule` commands and
the prescribed event heat-balance rule. Use the same `claim.freeze()` and
`FrozenController.from_json()` methods. `frozen.as_claim(tree=new_tree)` explicitly
rebinds held-out two-period paths while retaining the controller and physical
settings. Storage replay evaluates the entire bound tree and includes ordered
stage traces, carried energy and terminal checks. Missing preparation signals
are unresolved unless a fallback command was declared. See
[causal storage replay](storage-replay.md) for the scope and runnable example.

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
- Storage controllers: `StorageController`, `StoragePreparationRule`; portable
  replay uses the common `FrozenController` and engineering result types.
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
