# Package boundary: a process-engineering DSL

Design decision, 2026-09-11. The target responsibility of `operange`
is a **process-engineering sensitivity and robustness DSL**, with execution
through supported model backends and structured results for downstream code.
This note defines the intended boundary; it does not implement a new API.

## Core responsibilities

| Surface | Responsibility |
| --- | --- |
| Process model | Named inputs, design parameters, states, controls, observable outputs, equations, operating constraints, units, and applicability |
| Uncertainty | Admissible realizations, coupled sets, trajectories, faults, model alternatives, and membership evidence |
| Operating contract | Fixed operation, adjustable recourse, causal policies, observation timing, and initial/terminal-state rules |
| Sensitivity | Derivatives or directional responses of declared process outputs and constraints under an explicit evaluation contract |
| Robustness | Requirement audits, admissible breaking witnesses, supported boundary/severity searches, and evidence of scope and optimality |
| Engineering changes | Apply caller-supplied equipment or operating-contract changes and re-evaluate their physical consequences |
| Evidence | Typed quantities, residuals, bounds, solver status, model provenance, and portable model/operating-contract references where supported |

The DSL expresses the engineering question. Backend capabilities determine
which evaluations and guarantees are supported. The package need not become
a full thermodynamic simulator: analytical models, optimization models, and
external simulators can supply the physical evaluation behind the same
explicit contract.

## Downstream responsibilities

Applications own economics, accounting, tariffs and financial assumptions,
CAPEX/OPEX models, NPV, business-case predicates, attribution methods, repair
valuation and selection, business Pareto frontiers, report prose, charts,
dashboards, and integration-specific schemas.

The core returns the energy use, delivered heat, material flows, production,
state trajectories, and constraint evidence that those applications need.
It does not decide what an MWh is worth or which investment is preferred.
A change that preserves physical service can be audited independently of how
an application prices or ranks it.

Parameter metadata may preserve caller provenance, but does not make the
package responsible for its business interpretation. Domain-neutral objective
or expression support may let a caller supply coefficients or an output
functional that the backend can handle. Financial formulas, defaults, and
meaning remain in the caller. An arbitrary external callback does not acquire
global robustness guarantees merely by being attached to a process model.

Dependency direction is:

```text
downstream application
    -> process DSL and supported evaluation backend
    <- process values, sensitivities, robustness evidence
    -> downstream economics, attribution, selection, and presentation
```

Consumer-specific adapters belong to the consumer or a separate integration
package. The process package does not import a reporting application, embed
its wire format, or offer a presentation extra as part of its target scope.

## A sufficient output contract

Prefer a small set of typed, composable results rather than a ready-made
report. Proposed result groups are:

- **Evaluation:** named process quantities, units and axes, state/control
  values, feasibility status, constraint residuals, and model/solve references.
- **Sensitivity:** named input/output coordinates, evaluation point,
  perturbation and observation semantics, derivative operator or values,
  scaling, validity region, and numerical evidence.
- **Robustness:** the audited requirement, uncertainty domain, recourse
  contract, verdict, witness or boundary, available bounds, and coverage.
- **Change evaluation:** references to original and changed engineering
  contracts, input conditions, and their respective evaluations and audits.

Serialization of these values is an engineering interoperability concern.
Chart ordering, currency formatting, narrative explanations, and business
driver allocation are consumer concerns. Stable identities for constraints,
quantities, coordinates, and solves let consumers join evidence without
requiring the core to know their data model.

Unavailable, infeasible, unresolved, and unsupported results must remain
distinguishable. An unavailable derivative or failed model evaluation cannot
be encoded as a zero response. A consumer must be able to reproduce what was
established without parsing a human-readable message.

## Sensitivity is an explicit engineering query

The same parameter perturbation can have different effects under fixed controls,
a frozen feedback controller, and reoptimized recourse. A sensitivity result
must state which quantities are held fixed, which may adjust, and which
observations are available. It must also identify the parameterization and
units used for the perturbation.

Backend capabilities may include Jacobians, Jacobian-vector products,
vector-Jacobian products, local curvature, or finite-difference responses.
They need not all be available, and a dense Hessian is not a prerequisite for
using the package. Analytical, differentiated, and finite-difference results
retain their method and numerical qualifications. At an operating-mode or
active-set change, a one-sided/local result or an explicit unsupported status
is preferable to an unqualified smooth derivative.

Solver multipliers can be returned when their objective, residual orientation,
units, solve identity, and validity are known. Their interpretation follows
the objective actually solved. Maximizing delivered heat produces sensitivity
in heat units; no automatic conversion into money is implied.

For illustration, a consumer may define its own cost `C(y, p)` from process
outputs `y = F(theta)` and external economic inputs `p`. Where differentiable,
it can combine its derivatives of `C` with the process derivatives of `F`.
Any direct dependence of the cost on `theta` must also be included. The process
package supplies engineering responses; the consumer owns the financial
composition and any subsequent attribution. Higher-order composition likewise
requires the applicable chain-rule terms, not relabelling a physical Hessian
as a cost Hessian.

A local sensitivity is not a robustness certificate. Membership, recourse,
domain coverage, and global-bound obligations still apply to a breaking
witness. Keep these result types and their evidence separate even when they
come from the same backend.

## Implemented boundary and consumer examples

Markdown rendering, GBP-valued repair wrappers, CAPEX ordering, and selected-repair
helpers now live in repository consumer examples outside the process wheel.
Physical candidate application and full-domain re-audit remain in the core;
the consumer attaches costs, ranks candidates, renders results, and freezes
the engineering contract it chooses. Model-specific APIs are exposed through
`operange.reference`, separately from the shared top-level DSL.

The [experimental result contract](result-contract.md) now implements three
typed result families, with [worked cases](result-contract-cases.md) for
feasibility, sensitivities, robustness, and engineering changes. Structured
queries and portable exports are consumable without a renderer or monetary
field. Analytical heat-response first derivatives are implemented; a general
sensitivity DSL and broader numerical adapters remain future work.

Useful acceptance cases are the existing heat-capacity audit and storage
information conflict, plus a small process derivative with an independent
analytical reference. Each should return enough typed evidence for a consumer
to build its own explanation without moving its business model into the core.

The [API guide](../docs/api.md) covers engineering imports, and the
[getting-started guide](../docs/index.md) demonstrates downstream use.
The installed-wheel acceptance check runs a
public consumer outside the checkout with only declared runtime dependencies.
