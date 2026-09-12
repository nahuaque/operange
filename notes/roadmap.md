# Roadmap and open decisions

Part of the [process exploration](README.md). The static slice and a finite-tree
storage model now have an [implementation](../README.md).
General nonlinear adapters and production simulator integration remain future
work. Frozen affine controllers support static linear-model replay, and causal
storage controllers support ordered held-out paths. The legacy frozen heat
contract remains a separate static-recourse format. An experimental
[DWSIM connection](../docs/dwsim.md) now exercises a real installed simulator.

## Package boundary before expansion

The [scope decision](package-boundary.md) fixes the target as a process-engineering
sensitivity and robustness DSL. Its [result contract](result-contract.md)
now implements evaluation, sensitivity, and robustness outputs, validated against
the [worked acceptance cases](result-contract-cases.md). New structured APIs
sit alongside the prototype methods. The analytical heat response supports
Jacobian and directional queries with named physical or normalized coordinates;
unsupported response types and operators are explicit. Portable exports bundle
contracts and evaluations with evidence and content digests. Storage results
retain shared information-node controls and conflicting-future cardinality.

The next [shared DSL slice](dsl-primitives.md) is now implemented: named scalar
coordinate spaces, an uncertainty-domain protocol with box and generic finite
implementations, explicit distance and recourse primitives, and a common `Claim`
bound to a model adapter. Both reference models use these interfaces. Domain
operations and adapter proof capabilities are separate; unsupported bindings
do not inherit either reference model's numerical guarantees. The
[geometry slice](uncertainty-geometries.md) now implements labeled vectors,
polytopes, simplexes, budget sets, ellipsoids and typed composition. Membership,
normalization, nonemptiness witnesses and portable manifests are explicit;
the four numerical geometries additionally provide checked linear support bounds.
The [fixed affine adapter](affine-process-adapter.md) now exercises those
operations within the full claim/result contract, with direct evaluation,
ambient analytical sensitivities and support-bound audits. The utility example
holds the physical model constant while changing the uncertainty assumptions.
The [linear dispatch adapter](../docs/linear-dispatch.md) now adds bounded controls,
coupled equipment limits and finite-scenario audits under fixed or fully observed
static operation. The [failure-distance slice](../docs/failure-distance.md) adds
fixed affine threshold searches over boxes and polytopes, with exact checks of
physical candidates and bounded dual certificates across all selected requirements.
[Continuous recourse](../docs/continuous-dispatch.md) now covers boxes and
explicit convex hulls by exhaustive checked generators, with a caller-visible
enumeration budget. Other continuous geometries and distance searches with
adjustable controls remain future work.

Saved affine controllers now support continuous-domain audits through checked
linear bounds, including command rounding and every equipment bound. Boxes and
polytopes additionally support threshold-distance bounds with actual controller
replay. The linear point evaluator offers opt-in conflict reduction relative to
the declared control box and independently checked one-operating-limit relief.
The [controller example](../docs/frozen-controllers.md) connects a failing
continuous audit, adjustable point diagnosis, a caller-supplied equipment change,
full-envelope re-audit and frozen replay. These capabilities do not extend the
fully adjustable adapter's continuous-domain scope.

The [common engineering-change comparison](../docs/engineering-changes.md) now
re-audits caller-supplied claims across the shared and reference adapters. It
records equipment, operating, domain and service changes, preserves unresolved
audits and witnesses, and distinguishes restoration from a passing revised
commitment. Portable comparisons retain observed requirement margins and full
audit references; pricing and candidate selection remain consumer concerns.

The opt-in [quadratic/residopt experiment](quadratic-residopt-experiment.md)
now compares exact SDP compilation with an independent trust-region bound for
one fixed quadratic over an ellipsoid. Both feed the existing result contract.
The native checker is faster in the measured workload; residopt remains optional.
The [coupled design experiment](coupled-design-experiment.md) now runs in consumer
example code: three shared equipment sizes, three robust constraints, a material
objective, and independent feasibility/optimality checks. Residopt's compiled
and hybrid paths work; the native reference remains competitive. No general
design-optimizer API was added to the DSL.

The residopt investigation is parked. Prototype Markdown rendering, monetary
repair fields, CAPEX ordering, and selected-repair logic now live in repository
consumer examples. Engineering-change evaluation and re-audit remain in the
core. The [API guide](../docs/api.md) documents shared and reference
APIs and the versioned engineering result format. An isolated installed-wheel
acceptance check exercises
the [public consumer example](../docs/index.md), without workspace or
optional solver dependencies. Release versioning and publication remain separate.
The implementation slices below remain references for the engineering DSL.

## 1. Make the analytical example executable

Implemented, including analytical verification, repair comparison, and static
frozen-contract replay.

Start with the [heat-recovery example](heat-recovery-example.md), a small
realization schema, box membership, static recourse, normalized L-infinity
distance, a finite repair catalog, and a Markdown report. Keep the analytical
formulas independent of the numerical backend so they can check its results.

Acceptance conditions:

- Recover the feasibility-boundary radius `4/7` and the 10 kW breaking-witness
  radius `201/350`, with their different interpretations.
- Verify witness membership, physical balances, and impossibility of recovery.
- Rank all catalog repairs against the original domain and service requirement.
- Identify the source-heat limit and explain why compressor capacity alone
  does not fix it.
- Preserve units and distinguish original-claim repairs from reduced service.
- Show guarantee scope and numerical tolerances beside the result.

This first slice does not need a general bilevel solver, property package,
historian connection, or a large collection of uncertainty constructors.

## 2. Establish numerical evidence contracts

The two implemented models share an LP interface that validates inputs and
recomputes primal residuals. Independent analytical checks guard both verdicts
and objective values. Tests inject unresolved termination, false infeasibility,
and invalid optimal solutions; these remain inconclusive. The finite storage
tree is the first scenario-based model.

Implemented for bounded linear operation: `LinearProcessAdapter` reuses physical
affine outputs and adds `LinearControl` bounds plus always-applied operating limits.
Finite audits check every returned physical dispatch exactly or prove joint
infeasibility using nonnegative row combinations over the bounded control box.
Solver status alone cannot establish a verdict. The two-boiler example compares
fixed commands, redispatch and a larger shared fuel supply under the same finite
loads. Tests include false infeasibility, invalid primal and dual candidates,
extreme scaling, selected requirements and partial coverage with a valid witness.
General nonlinear and surrogate evaluations remain separate future capabilities.

Add a nonlinear local-search example with a known recoverable realization
where a poor solve can fail. The required outcome is `inconclusive`, unless
other evidence resolves it; it must never become a certified breaking witness
merely because the attempted solve failed.

Only claim minimum distance when the outer search also has the needed global
bounds. Convex inner recourse alone does not provide that guarantee.

## 3. Validate the trajectory contract

The [two-period storage tree](thermal-storage-example.md) is implemented. It
demonstrates fixed-policy, causal-policy-class, and perfect-foresight outcomes
separately, including shared preparation decisions, state continuity, terminal
conditions, and a minimum pair of individually feasible but incompatible futures.
The repair comparison distinguishes storage capacity, charging power, and an
explicit earlier-information intervention. Evidence covers the declared finite
tree; it does not establish robustness over continuous or unlisted trajectories.

The [frozen-controller slice](../docs/frozen-controllers.md) now captures a linear
model, explicit affine controller, domain, permissions and evidence settings.
Direct replay retains controller-specific failures and supports new finite
scenario domains without retuning. Continuous controller audits and bounded
threshold searches are also supported as described above. It integrates with
the common comparison API.

The [causal storage replay slice](../docs/storage-replay.md) now freezes the
two-period storage model, tree, explicit controller, initial-state rule and
evidence settings. Ordered held-out paths preserve observation timing and carry
energy from preparation into the event without retuning. The round-trip manifest
resolves the executable model and controller. Each path starts from the declared
initial energy; continuous multi-event history and general feedback policies
remain future work.

## 4. Attempt one real simulator adapter

The [startup-profile slice](../docs/startup.md) now accepts externally
supplied piecewise-linear load profiles, fixed start commands and finite
amplitude/duration/timing scenarios. It checks shared peaks and optional time
integrals over all declared time segments and compares staggered schedules.
This is a bounded transient response model, not a general trajectory domain or
simulator integration. General dynamics and continuous uncertainty remain below.

The [DWSIM connection experiment](../docs/dwsim.md) now runs the bundled Humid Air
sample through DWSIM 10.2.6 on macOS ARM64. Four finite cases vary feed temperature
and flow under its fixed cooler specification, return fresh stream results, and
close mass and energy balances. The experiment uses the desktop's IronPython
script manager and a checked JSON batch exchange; it is not yet a public adapter.
The next slice needs an isolated headless worker, explicit model and settings
provenance, and fixed-operation evaluation plus finite audits. Nonconvergence
must remain unresolved. Production macOS automation compatibility is still open.

Measure model evaluations, runtime, unresolved runs, verified violations, and
the fraction of candidates for which recovery can actually be ruled out.
If evidence supports only candidate failures, the report should say so.
That outcome may still be useful, but would change the initial product promise.

## Later extensions

Add trajectories from historical data, degradation with maintenance decisions,
fault combinations, and model-form
alternatives when a concrete example demands them. Keep probability-law
ambiguity as a distinct later track with its own claims and witness semantics.

Investigate backend reuse before writing new optimization machinery; see the
references and restrictions in [Evidence and backends](evidence-and-backends.md).
Extract shared engineering/evidence infrastructure only when concrete models
demonstrate the need. Report infrastructure remains downstream.

## Decisions to revisit after the first example

| Question | Working assumption | Evidence that would change it |
| --- | --- | --- |
| Repository and distribution | Standalone Operange project with an engineering-only scope | Revisit only if shared infrastructure becomes necessary |
| Main claim type? | Static capacity and a finite-tree causal service claim | A real use case requires a broader model or frozen controller |
| Closest failure or robustness radius? | Show both when the same geometry supports them | Domain topology, discrete modes, or solver cost makes one misleading |
| Distance defaults? | User-declared engineering scales | A well-validated historical model supports a different severity measure |
| Meaning of "repair"? | Audit caller-supplied changes preserving the service claim; consumers value and select them | Design-dependent uncertainty requires a broader engineering contract |
| First simulator? | DWSIM, beginning with the bundled Humid Air connection experiment | Headless runtime availability or evidence limitations prevent a useful fixed-operation adapter |
| Evidence threshold for shipping? | Every verdict explains what was actually established | User studies show a different artifact is needed to support decisions |

Success means downstream code can evaluate a process response, inspect its
sensitivities, reproduce a failure, and re-audit engineering changes using
structured evidence. That is the criterion for deciding which abstractions
deserve to become public API.
