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
| Connected steady-state equipment | `ProcessNetwork`, `ProcessStream`, `StreamSource`, `StreamSink`, `Mixer`, `UtilityHeader`, `FixedRatioSplitter`, `Boiler`, `StreamTotal` |
| Dispatch objectives | `LinearObjective`, `ControlTrackingObjective`, `ControlTarget` |
| Optimized objective sensitivities | `DispatchChange`, `DispatchSlope`, `DispatchSensitivity`, `DispatchSensitivityResult`, `dispatch_sensitivity` |
| Controller synthesis | `ControllerSynthesis`, `synthesize_controller` |
| Independent certificate checks | `verify_result`, `CertificateVerification`, `VerificationCheck` |
| Executable linear controllers | `AffineControlRule`, `AffineController`, `FrozenController` |
| Measurement uncertainty | `AffineMeasurement`, `MeasurementModel` |
| Guaranteed operating ranges | `OperatingLimitResult`, `OperatingLimitTrial`, `operating_limit` |
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

`claim.operating_limit("production", tolerance=1e-6)` searches nested ranges of
one scalar coordinate in a box under fixed commands, adjustable linear dispatch,
or a saved affine controller. Its separate `operating_limit/v1` report retains
checked achievable/excluded endpoints and physical audits. A passing search
ceiling is not a global maximum; unresolved audits never establish an upper
bound. Read [guaranteed operating limits](operating-limits.md) for units,
coverage, replay and independent verification.

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

`claim.dispatch_sensitivity(realization, changes=(DispatchChange(...), ...))`
checks one-sided optimized-objective slopes for physical inputs, operating
limits and control bounds. Exact primal/dual paths establish local validity
intervals; kinks and infeasible directions remain explicit. The separate
`dispatch_sensitivity/v1` report supports portable independent verification.
See [dispatch and bottleneck sensitivities](dispatch-sensitivity.md).

`ProcessNetwork.compile()` generates a `LinearProcessAdapter` from connected,
unit-compatible material or thermal-power streams. Physical balances always
apply; service sinks remain selectable requirements. Its `process_network/v1`
source JSON is separate from compiled results and frozen controllers. See
[equipment and stream modelling](equipment-networks.md) for declarations,
equations, algebraic rate controls and the two-boiler and water-mixing examples.

`verify_result(exported_json, expected_contract_id=...)` independently checks
supported linear dispatch, objective, relief and controller-audit certificates
without invoking a solver or model query. It accepts standard and compact
result exports; `python -m operange.verify result.json` provides a command-line
interface. Unlike `result_from_json`, it reconstructs physical rows and rechecks
the mathematics. See [certificate verification](certificate-verification.md)
for supported proofs, limits and explicit unsupported results.

## What the built-in adapters establish

The optional [IDAES integration](idaes.md) exports `IdaesAdapter`,
`IdaesVariable`, `IdaesRequirement` and `IdaesLinearExport` from
`operange.integrations.idaes`. Its native results report numerical point
feasibility and finite-scenario coverage, with unresolved outcomes for failed
solves. They carry no nonlinear infeasibility or independent certificate claim.
Supported exact affine exports use the existing linear model capabilities below.

The public [calibration integration](calibration.md) exports `ParmestCalibration`,
`CalibrationResult`, `CalibrationEstimate`, `CalibrationObservation`,
`CalibrationCheck` and `CalibrationExperimentResult` from
`operange.integrations.parmest`. It delegates shared-parameter fits and
reconciliation to upstream parmest, checks returned candidates, exports
`calibration_result/v1` records and applies completed estimates to fresh
`IdaesAdapter` snapshots through an explicit parameter map. Running fits needs
the `idaes` extra and Ipopt; importing declarations and reading results does not.
`ParmestCalibration.bootstrap(samples=..., seed=...)` additionally returns a
`CalibrationBootstrap` containing joint `CalibrationDraw` records, checked
physical replays and empirical covariance. It exports `calibration_bootstrap/v1`,
provides a joint finite domain, and builds finite IDAES claims through explicit
parameter mapping and dispatch permissions. See [joint propagation](joint-calibration-uncertainty.md).
Identifiability and statistical confidence coverage are not assessed.

`ParmestCalibration.diagnose(fitted, parameter_scales=...)` returns
`CalibrationDiagnostics` with local scaled measurement Jacobians and weak
parameter combinations. `ParmestCalibration.validate(fitted, experiments=...)`
returns `CalibrationValidation` with predictions for explicitly held-out
experiments, without refitting their measurements. The accompanying
`CalibrationSensitivity` and `CalibrationPrediction` records are exported from
`operange.integrations.parmest`. Portable schemas are
`calibration_diagnostics/v1` and `calibration_validation/v1`; summaries require
only base dependencies. These reports do not establish structural
identifiability or statistical confidence. See
[calibration credibility](calibration-credibility.md).

`IdaesAdapter.optimize_shared_controls(finite_domain)` chooses one command vector
using the worst-case active source objective. The calibration convenience method
`CalibrationBootstrap.optimize_shared_controls(adapter, operating_cases,
parameter_map=...)` preserves whole draws and crosses them with operating cases.
Both return `IdaesSharedControlResult` (exported by `operange.integrations.idaes`),
with source-checked scenario evidence and portable `idaes_shared_controls/v1`
serialization. Completion establishes finite numerical feasibility; global
optimality and infeasibility remain uncertified. See
[shared IDAES commands](shared-idaes-controls.md).

`IdaesAdapter.propagate_uncertainty(parameters=..., covariance=..., ...)` computes
local output and signed-margin Jacobians and covariance. Supply every fixed
actuator command, or explicitly request `reoptimize=True` with native-unit
perturbation steps. `CalibrationBootstrap.propagate_uncertainty(adapter,
parameter_map=..., ...)` supplies the full empirical covariance at the checked
fit. Both return `IdaesUncertaintyResult`, exported alongside
`IdaesUncertainParameter`, `IdaesResponse` and `IdaesPerturbation` from
`operange.integrations.idaes`. The `idaes_local_uncertainty/v1` reader and
summaries need only base dependencies. This separate local diagnostic does not
upgrade generic claim sensitivity or certify robustness. See
[local operating uncertainty](local-uncertainty.md).

`ParameterPoint`, `ParameterBinding`, `ParameterCovariance` and `ParameterSamples`
are provider-independent records exported from `operange.parameters` and
`operange`. `CalibrationResult.parameter_point` and
`CalibrationBootstrap.parameter_samples` expose checked data in this form.
`IdaesAdapter.with_parameters(point, parameter_map=...)` binds external values
on a fresh snapshot; local propagation also accepts named covariance records.
See [parameter data and execution](parameter-data-and-execution.md).

| Adapter | Evaluation and sensitivity | Robustness |
| --- | --- | --- |
| `AffineProcessAdapter` | Fixed affine responses; analytical first derivatives in physical or normalized coordinates | Finite and supported linear-support audits; L∞ box/polytope distances by default, and optional convex-domain L∞/Euclidean distances; no adjustable recourse |
| `LinearProcessAdapter` | Joint bounded-control feasibility; opt-in irreducible row conflicts and single or jointly weighted operating-limit relief via `evaluate_result(diagnose=True, relief=...)`; separate checked objective-value slopes via `dispatch_sensitivity` | Complete `FiniteSet` audits, plus continuous `BoxSet`/`ConvexHullSet` audits by checked generators under fixed or fully observed static operation; exact physical feasibility and infeasibility certificates; no adjustable distance searches |
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

`claim.audit_result(relief={"changes": [...], "objective": "linear"})` finds one
shared operating-limit relief plan across a `FiniteSet`, `BoxSet` or
`ConvexHullSet`. It preserves the original audit verdict and adds checked
equipment-objective bounds plus a full-domain re-audit of any candidate.
Quadratic relief requires `backend="cvxpy"`; `relief.max_scenarios` defaults to
256. Service requirements, hard control bounds and recourse permissions remain
fixed. See [shared equipment relief](shared-relief.md) for declarations and replay.

`claim.synthesize_controller(objective=LinearObjective("fuel"))` chooses permitted
affine coefficients over boxes, finite sets and explicit convex hulls using the
optional CVXPY backend. Boxes with a linear objective use whole-domain support
constraints without corner enumeration; finite sets, hulls and quadratic box
tracking use complete generators subject to `max_vertices`. The model's
objective is used when omitted. A separate
`ControllerSynthesis` report retains the candidate, its independent rounded-command
audit and whole-domain performance enclosure. `search.freeze()` exports an
accepted candidate through `FrozenController`. Numerical failure establishes no
adjustable infeasibility, and controller optimality is not certified. See
[controller synthesis](controller-synthesis.md) for permissions, tracking and scope.

`LinearProcessAdapter(measurements=MeasurementModel(...))` gives controllers
named affine sensor readings over joint true-input and sensor-error uncertainty.
Permissions reference readings; physical equations retain true inputs. Synthesis,
frozen replay and continuous audits account for sensor and command rounding.
Adjustable evaluation requires a bound controller to preserve the information
restriction. See [measurement uncertainty](measurement-uncertainty.md).

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
