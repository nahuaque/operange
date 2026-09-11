# Evidence, backends, and the frozen contract

Part of the [process exploration](README.md). These are proposed semantics,
not guarantees supplied by the current package.
The [package boundary](package-boundary.md) assigns these mathematical and
engineering contracts to the core; consumers own economic interpretation,
candidate valuation, and report presentation.

## Claim verdict and evidence are separate fields

Keep `pass`, `fail`, and `inconclusive` as claim verdicts. Record the evaluated
domain and control contract alongside the verdict. A finite-scenario claim can
pass after exhaustive evaluation; evaluating sampled scenarios from a larger
continuous domain cannot establish that larger universal claim.

Evidence needs multiple fields rather than one ordered badge:

| Field | Examples |
| --- | --- |
| Domain coverage | Complete finite set; continuous set with verified bound; sampled subset |
| Recourse scope | Frozen policy; unrestricted static recourse; affine policy class; causal tree policy |
| Solve evidence | Analytical proof; global bounds; local optimum; feasible point; time limit |
| Model basis | Declared physical model; simulator version; surrogate with or without error bounds |
| Numerical qualification | Residual tolerances, constraint scaling, objective gap, solver status |
| Statistical qualification | Calibration assumptions and coverage statement, if separately justified |

A globally solved surrogate still has surrogate-level model validity. An
exact result on a finite scenario set still has finite-set scope. Both may be
valuable, but neither warrants an unqualified physical robustness certificate.

For the full-domain claim, use `inconclusive` when search finds no failure and
cannot bound the unexplored domain. Retain the useful result as "no violation
found in N evaluated scenarios", with failures to evaluate counted separately.

## A breaking witness has two proof obligations

First establish that the disturbance belongs to the declared uncertainty set.
Then establish that it breaks the exact control contract being audited.

For static adjustable operation, define a normalized violation value

$$
v(d,u,m)=\inf_{(x,z)\in F(d,u,m)}
\max_j\frac{g_j(d,x,z,u,m)}{s_j},\qquad s_j>0.
$$

The scales carry the units of their constraints. Assume the inner infimum is
attained for this interpretation; otherwise attainment and boundary behavior
need explicit treatment. Set the value to positive infinity for a **proven**
empty feasible set. A nonempty feasible set with `v <= 0` can meet the claim.

A feasible control gives an **upper** bound on this minimization. A positive
objective from a local solve therefore does not prove that recovery is
impossible. A positive valid **lower** bound, analytical reasoning, or an
appropriate infeasibility certificate is needed to establish failure under all
allowed controls. Failed convergence gives no such certificate.

For a frozen policy, a valid model evaluation with a violated requirement can
refute that policy directly. If model states are nonunique, the adapter must
specify how solutions are selected or quantified over; a solver's arbitrary
state selection cannot silently define the claim.

For a searched causal policy class, failure may require a branching
counterexample, not one trajectory. Two individually recoverable futures can
require conflicting actions before their histories diverge. The result should
retain that shared-history conflict or its infeasibility certificate.

## Minimum severity is an additional claim

For the static contract, the intended inverse problem is

$$
\rho_\epsilon=\inf_{(u,m)\in\mathcal U}
D((u,m),(u_0,m_0))
\quad\text{subject to}\quad v(d,u,m)\ge\epsilon,
\qquad\epsilon>0.
$$

The positive separation must exceed the applicable numerical uncertainty.
Different requirements may instead use explicit margins in their original
units. Equality at a passing threshold is a boundary point, not a strict
failure; report boundary distance and margin-separated witness separately.

Every verified failing candidate gives an upper bound on minimum breaking
distance. Only a global exclusion of closer failures gives a lower bound.
Report `[lower_bound, upper_bound]`, the gap, and the candidate severity; use
"minimum" only with the corresponding proof. A nominal failure has distance
zero when the baseline is admissible. Report "no breaking realization exists"
only when supported over the full declared domain.

Use normalized L-infinity for the first example. Later distances can include
L1, Mahalanobis, and caller-defined severity where the backend supports it.
Any economic conversion belongs to the caller. Covariance regularization and
reference scales must be frozen and recorded. Discrete faults can use a count or a
lexicographic objective such as `(fault_count, continuous_severity)`. Avoid
adding pounds, temperatures, and fault counts without an explicit conversion
or preference rule. Distances need not be probabilities or mathematical metrics.

## Capability-driven backend boundary

```text
claim + uncertainty + model + recourse + distance
    -> validate units, domains, information pattern, and capabilities
    -> select an applicable evaluation/search backend
    -> retain candidate points, bounds, residuals, and unresolved evaluations
    -> apply claim and witness evidence rules
    -> return typed results and provenance to the caller
```

| Backend | Useful first responsibility | What it must not infer |
| --- | --- | --- |
| Analytical toy | Closed-form recourse and witness; reference values | Applicability to a richer plant model |
| Finite enumeration | Check all declared scenarios/modes using valid recourse evidence | Coverage of an unstated continuous domain |
| LP / convex solver | Supported inner recourse and certified reformulations | Generic tractability of nested adversarial search |
| Nonlinear/local search | Produce candidate controls and disturbances | Global infeasibility from local termination |
| Simulator adapter | Evaluate model state, residuals, and failure candidates | Plant infeasibility from a timeout or nonconvergence |
| Surrogate search | Propose informative candidates cheaply | Original-model validity without verification or error bounds |

The adapter should return quantities with units, named constraint residuals,
state/control values, model identity, runtime, and structured solve status.
Distinguish invalid external input, unresolved model evaluation, certified
infeasibility, and a feasible operation that misses the claim. Unsupported
capabilities should produce a diagnostic, not silently downgrade a requested
certificate to sampling.

Existing work supports this separation of modeling and solving. The
[ROmodel paper](https://link.springer.com/article/10.1007/s11081-021-09703-2)
describes uncertain parameters, uncertainty sets, adjustable variables, and
solver applicability restrictions. Its ability to represent a set does not
imply an applicable solution method. This motivates explicit capabilities here.

[PyROS](https://pyomo.readthedocs.io/en/stable/explanation/solvers/pyros/index.html)
is an existing Pyomo solver for nonlinear two-stage adjustable robust
optimization. It is a candidate for a later adapter investigation, not a
dependency or a claim that all proposed process contracts fit its methods.

## Repairs must retain the original claim

A core engineering change has a declared effect on design or operating
permissions, physical downtime, implementation constraints, and provenance.
Its financial cost and business valuation belong to the consumer. Re-audit the
entire original uncertainty domain after applying it. Removing the displayed
witness alone is a diagnostic, not a robustness repair certificate.

Return each candidate's engineering result, domain coverage, and unresolved
evaluations. A downstream application can attach costs and select the cheapest
verified passing candidate in its catalog. A global economic optimum needs its
own search evidence; the core audit does not establish that claim. The current
prototype's monetary fields and ranking helpers are conveniences scheduled for
separation from the target DSL.

Reducing required production, allowing a new shutdown mode, or excluding
formerly admissible weather changes the claim or domain. Show those as revised
contracts alongside repairs. If a design changes the uncertainty itself,
represent `U(d)` explicitly and explain the mechanism; never shrink the set
silently to make the design pass.

## Freeze the contract, then evaluate later evidence

A proposed `freeze()` manifest should retain:

- claim predicates, units, aggregation periods, scales, and separation margins;
- design, engineering change identity, and the controller or recourse policy class;
- model, property package, simulator, and adapter versions and content hashes;
- uncertainty definitions, fault budgets, distance, baseline, and data lineage;
- observation timing, grid, initial/terminal-state rules, and reset permissions;
- calibration/training cutoff and historical preprocessing;
- solver settings, tolerances, seeds, bounds, and evidence artifacts;
- schema version and fingerprint over the evaluable contract.

Portable freezing needs versioned model/policy references and resolvable
artifacts. A Python lambda or arbitrary simulator session cannot be made
reproducible merely by putting its display name into JSON. A fingerprint
identifies content; it does not authenticate the model or prove its accuracy.

Holdout replay must use the declared controller and only information available
at each historical decision time. A period-by-period perfect-foresight optimum
can be reported separately. Define whether state carries across periods or
resets according to a frozen rule; do not use later outcomes to initialize it.

Report observed requirement violations, excursions outside the frozen domain,
unresolved evaluations, and model applicability drift separately. A calibrated
radius breach is a review trigger. Recalibration, redesign, or controller
retuning creates a new contract version rather than revising the holdout test.
