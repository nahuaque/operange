# Engineering result contract: v1

Implemented experimental slice, 2026-09-11. The
[package API](../README.md#public-api-and-evidence-scope)
now supplies the three immutable result families below as `process_result/v1`,
with self-contained JSON bundles and validated references. Heat recovery supports
evaluation, analytical optimized-response first derivatives, full-box audits,
and boundary/breaking searches. Storage supplies joint finite-tree evaluation
and robustness results. Broader operators and adapters described below remain
future work; unsupported capabilities are explicit results.

The public result family has three members: `EvaluationResult`,
`SensitivityResult`, and `RobustnessResult`. They share quantity, identity, and
evidence conventions. An engineering-change comparison composes these results;
it does not need a fourth result family or an investment-selection abstraction.

## Shared envelope

| Field | Meaning |
| --- | --- |
| `schema_version` | Experimental wire identity `process_result/v1` |
| `kind` | `evaluation`, `sensitivity`, or `robustness` |
| `result_id` | Stable identity of this immutable result artifact |
| `contract_ref` | Identity and content digest of the engineering contract used |
| `request` | The requested operation, selected quantities/coordinates, evaluation scope, and operation-specific settings |
| `execution` | `completed`, `unresolved`, `unsupported`, or `invalid` |
| `payload` | The result-specific fields defined below |
| `evidence` | Structured checks and referenced numerical/proof artifacts |
| `diagnostics` | Stable codes, affected field/quantity references, and optional human-readable context |
| `artifacts` | Bundled `contract` and supporting `evaluations`; loading validates records without executing a solver |

`completed` describes execution of the requested calculation. It does not mean
that a service requirement passed, a global optimum was found, or an entire
uncertainty domain was covered. For example, a completed sample audit can have
an inconclusive universal verdict. Useful verified evidence can survive an
unresolved search. Consumers branch on the semantic fields as well as execution.

Invalid inputs and unsupported capabilities leave the semantic result
`not_evaluated` or `not_assessed`. They do not become physical failures. Unknown
values are not zero, and explanations are never the only place a status lives.

The referenced engineering contract contains the versioned model and its
applicability, design, quantity/coordinate definitions, uncertainty domain,
requirements, operating permissions, objective/solution-selection rule, and
numerical policy. Dynamic contracts also include time axes and durations,
observation histories, and initial/terminal-state rules. Result-specific
realizations and query settings belong in `request` and are retained too.

A reference contains an artifact identity, schema/model version where relevant,
and content digest. A display name or local path alone is insufficient. A
portable export bundles all required manifests and evidence, or explicitly
declares external dependencies and how to resolve their immutable versions.
Runtime/model availability remains separate from being able to read the result.

## Quantities, coordinates, and evidence

`QuantityValue` refers to a named quantity in the contract registry and carries
a finite scalar or array. The registry declares unit, physical meaning, role
(input, design, state, control, output, or residual), and named axes. For example,
electrical power and heat flow both use MW but have different quantity identities.
Array dimensions and coordinate keys must match the registry; positional
coincidence is not an alignment rule. Scalar values have no axes.

Time-dependent rates retain period durations and aggregation semantics. Scenario
values retain scenario/node identities. A causal preparation control belongs to
its shared information node, with scenario references pointing to that control;
duplicating it into apparently independent per-scenario decisions loses meaning.

The request identifies physical or normalized input coordinates. If
`z_i = (theta_i - origin_i) / scale_i`, retain `origin_i`, positive `scale_i`,
and units. A severity scale and a derivative perturbation step are different
settings. Derivatives in normalized coordinates cannot be labelled as
derivatives in physical units without the corresponding transformation.

Evidence records identify:

- the proposition/check, its subject, method, and verification outcome;
- model/solve identity and backend/version/settings;
- observed residuals, reference errors, or objective bounds, with units;
- tolerances and the coordinates in which they apply;
- required assumptions and references to independent checks or certificates.

Use separate residuals for physical equations, hard operating limits, and
service requirements. Canonical inequalities have residual `r <= 0`; equalities
have `r = 0`. Accept them within their declared tolerances. If a positive-is-good
margin is also exposed, register it as a separate quantity with its sign relation.
Residual orientation is also part of any reported multiplier's contract.

Bounds are records with `state = finite | unbounded | unknown`, a finite value
when applicable, units/coordinate references, and supporting evidence. An absent
bound means unknown. Zero is a bound only when justified. Do not serialize NaN
or infinity as ordinary measurements; an infeasible set can instead have an
explicit infeasibility result.

## EvaluationResult

The request selects a realization or declared tree, an operating contract,
and outputs. It also states whether it evaluates prescribed operation, seeks
feasible operation, or optimizes a declared process quantity.

| Payload field | Meaning |
| --- | --- |
| `feasibility` | `feasible`, `infeasible`, `unknown`, or `not_evaluated` for the exact constraint set requested |
| `membership` | `inside`, `outside`, `unknown`, or `not_checked`, with a domain reference and evidence |
| `values` | Available named physical quantities; no fabricated operating point after an infeasible solve |
| `constraint_checks` | Hard-model residuals and, when requested, individual service-requirement assessments |
| `objective` | Optional: quantity, sense, attained value, optimum bounds, attainment, and scope of optimality evidence |
| `operating_solution_ref` | Optional reference to the evaluated controls/policy and state solution |

For every constraint, the contract distinguishes model feasibility from a
service requirement under audit. A requirement assessment can be `satisfied`,
`violated`, `unknown`, or `not_evaluated`. A feasible operating point may violate
a service requirement. A violated requirement at one attempted operation does
not show that every allowed operation fails.

If the feasibility request includes the service requirements as constraints,
an infeasibility certificate covers that augmented constraint set. It must not
be described as failure of the physical equations alone. The result identifies
the included constraint IDs.

For an objective, always define bounds as `lower <= optimum <= upper`, regardless
of optimizer sign conventions. A feasible point provides a lower bound for a
maximization and an upper bound for a minimization. Calling a quantity “maximum
delivery” requires the upper-bound evidence too; a merely feasible delivery
retains that weaker meaning.

`infeasible` requires appropriate infeasibility evidence. A timeout, nonconvergence,
or unsuccessful local attempt gives `unknown`. Availability of a feasible point
and certification of its optimality are independent fields.

The initial contract preserves the prototype's requirement that evaluations
belong to the declared domain. An outside realization is an invalid request;
unresolved membership blocks a verified in-domain result. A caller can create a
separate contract for exploratory evaluation without widening a frozen domain.

## SensitivityResult

The request references a base evaluation, selected input and output coordinates,
the derivative requested, and the allowed response of controls and states.
It must distinguish:

- **fixed controls:** hold the specified controls fixed and resolve permitted states;
- **frozen policy:** evaluate the same controller with its declared observations;
- **optimized response:** differentiate a declared optimum or solution map,
  retaining the objective, observation permissions, and solution-selection rule.

| Payload field | Meaning |
| --- | --- |
| `availability` | `available`, `undefined`, `unknown`, or `not_evaluated` for the requested derivative |
| `base_evaluation_ref` | Exact point/solution and contract being differentiated |
| `response_contract` | Fixed quantities, adjustable quantities, objective/policy, and observation timing |
| `derivative` | Requested operator, named input/output coordinates, representation, values, and units/scales |
| `method` | Analytical, differentiated, or finite difference, with required settings |
| `validity` | Point/local region, active-set or mode qualifications, one/two-sided interpretation, and supporting evidence |

Start with a materialized Jacobian and a directional derivative. A Jacobian
entry has the output's unit divided by its input coordinate's unit. For a
directional request `theta(t) = theta0 + t*v`, `t` is dimensionless, `v` carries
each input unit, and the result has the output unit. Preserve `v` in the result.
Derivative values align to explicit row/column coordinates, not display order.

An available derivative covers the whole requested operator. Partial usable
blocks may be retained as separately scoped results; a missing matrix entry
cannot silently become zero. Finite differences preserve stencil, steps,
evaluation references, residuals, and error/stability checks. Requesting an
analytical derivative does not silently fall back to finite differences.

`undefined` is a mathematical conclusion about the requested derivative, for
example a two-sided derivative at a known kink. `unknown` means the method
could not establish it. A one-sided or directional derivative can still be
available under a different explicit request. Differentiating a nonunique
optimizer requires a selection rule; the optimal scalar value may have a
derivative even when its optimizing controls do not.

A local region is declared and justified, or absent. A zero Hessian within one
operating regime does not imply global linearity. Vector-Jacobian products,
Hessians, and live operator handles are later capability extensions. A consumer
can receive an explicit unsupported result for them in the first implementation.
The wire contract initially carries materialized numeric data, not serialized
Python callables.

## RobustnessResult

The request identifies the claim, uncertainty domain, and recourse contract.
Its operation is `audit`, `minimum_breaking_severity`, or `boundary_search`.
Finite-tree causal claims and fully observed static claims retain their
different quantifier and information contracts.

| Payload field | Meaning |
| --- | --- |
| `verdict` | `pass`, `fail`, `inconclusive`, or `not_assessed` for the original full-domain claim |
| `coverage` | Domain reference, coverage method, evaluated support, unexplored support, and coverage evidence |
| `evaluation_refs` | Supporting engineering evaluations and policies |
| `witness` | Optional validated realization, path, or incompatible-futures witness |
| `search` | Optional target, candidate, certified severity bounds, attainment, and minimum-resolution evidence |

Coverage methods include complete finite enumeration, an analytical/global
argument over a continuous domain, and partial sampling/search. Counts alone
never establish continuous-domain coverage. A sampled verified failure can
refute the full claim; a sampled collection of passes cannot establish it.

Witness validation must establish both membership and failure under the declared
recourse. A fixed-policy counterexample refutes that policy. An adjustable-policy
counterexample needs evidence that the permitted recovery cannot meet the claim.
A causal witness may instead be a set of admissible futures and the shared
information node whose actions cannot satisfy them together.

The witness carries its kind, referenced realizations/nodes, affected constraints,
and membership/recourse evidence. Include conflict inequalities and named state
quantities when available. Candidate failures awaiting either check remain in
search diagnostics and are not placed in the validated `witness` field.

Search has its own `resolution`:

| Resolution | Meaning |
| --- | --- |
| `minimum_verified` | A verified attained candidate and global lower bound agree within the declared severity tolerance |
| `bounded` | Some valid bounds/candidate exist, but minimum or attainment remains unestablished |
| `unreachable` | The requested search target is excluded over the whole declared domain |
| `unresolved` | No supported search conclusion; retain useful partial evidence |

Keep `target_kind`, comparison relation, physical shortfall margin, severity
definition, reference point, and normalization explicit. For minimizing
severity, a verified target-reaching candidate gives an upper bound; excluding
closer candidates gives a lower bound. Equal bounds alone do not establish an
attained minimum if the infimum is unattained.

A boundary point at an inclusive requirement is not a breaking witness. A
`boundary_search` may therefore return `minimum_verified` with `verdict =
not_assessed`; a separate audit supplies any full-domain verdict. If a target
is a nonpositive-margin set, preserve that exact relation: a zero-distance
result at an already-failing nominal point is not a newly located zero-margin
boundary.

A verified breaking witness can give `verdict = fail` while its minimum search
remains `bounded` or `unresolved`. Conversely, `unreachable` for a requested
10 kW shortfall does not prove absence of smaller violations. It does not
automatically set `verdict = pass`.

Minimum cardinality of a conflicting set of futures is a different claim from
minimum disturbance severity. Store cardinality evidence on the witness with
its criterion and bounds; leave continuous severity unrequested. Do not turn
“two futures suffice” into a robustness radius.

## Consumer rules and versioning

Consumers can join outputs and construct explanations using IDs, typed values,
and evidence. They must not infer robustness from execution status, feasibility
from objective size, optimality from a solver message, or a radius from a local
derivative. Unsupported evidence limits the result even if a consumer can still
display a useful partial calculation.

Applying an engineering change produces a new immutable contract and new results.
The caller links before/after results and checks which domain, requirement, and
information references were preserved. No cost field, candidate ranking,
selected action, chart component, or narrative template belongs in this contract.

For a later v1 release, reject unknown major versions, unknown semantic enum
values, mismatched references, duplicate coordinate identities, and incompatible
units/shapes. Readers may ignore additive optional metadata in a compatible
minor revision, but must reject unsupported required capabilities. Freeze
canonical contract content separately from presentation and transient runtime
timing. Digests establish content identity, not mathematical validity.

The first implementation should validate these three result families and
preserve the two existing analytical examples. It need not introduce a schema
registry service, a general model compiler, a renderer, or a new dependency to
demonstrate the boundary. See the [worked contract cases](result-contract-cases.md)
for expected values, status combinations, and migration decisions.
