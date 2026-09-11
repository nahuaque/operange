# Compare engineering changes

`compare_changes` re-audits a baseline and named candidate claims through the
same API. It preserves each candidate's full audit, identifies changed
declarations, and distinguishes restoring the original service from changing
the service or uncertainty assumptions. It preserves caller order and leaves
costs, ranking and selection to the consumer.

## A boiler example

The baseline boiler can deliver 18 MW. Its fuel supply allows 25 MW of fuel at
80% efficiency. The finite demand envelope contains 16 MW nominal and 20 MW peak
steam loads. These are synthetic engineering values.

```{literalinclude} ../examples/engineering_changes.py
:language: python
:start-at: from dataclasses import replace
:end-before: if __name__
```

Run the example from the checkout:

```bash
uv run python -m examples.engineering_changes
```

The common comparison reports:

| Candidate | Audit | Commitment | Transition |
| --- | --- | --- | --- |
| Larger boiler: 20 MW | Pass | Preserved; equipment changed | `restored` |
| More fuel: 26 MW | Fail | Preserved; equipment changed | `still_failing` |
| Exclude peak demand | Pass | Revised domain | `not_comparable` |
| Permit 2 MW shortfall | Pass | Revised service | `not_comparable` |

Extra fuel alone cannot overcome the boiler's steam limit. Removing the peak
case or allowing a shortfall makes a different claim pass. Each row retains
its actual verdict and evidence, including rows that cannot establish recovery
of the original service.

## Use the common API

Import `EngineeringChange`, `compare_changes` and `ChangeComparison` from
`operange`. An `EngineeringChange(name, claim, provenance)` supplies an already
built common `Claim` and a nonempty explanation of the proposal's origin.
Names must be unique within a comparison.

`compare_changes(baseline, changes)` and `baseline.compare_changes(changes)` are
equivalent. They validate the inputs before auditing, audit the baseline once,
and audit each candidate once in caller order. They do not modify the supplied
claims. An empty sequence still returns the baseline audit. Convert heat and
storage reference claims with `.as_claim()` first.

`comparison.baseline` and `candidate.audit` are existing `RobustnessResult`
objects. Inspect their `execution`, `payload.verdict`, `payload.coverage`,
`payload.witness`, diagnostics and supporting evaluations. An unsupported or
inconclusive audit remains visible. An adapter error during an audit becomes an
unresolved row with a `comparison_audit_error` diagnostic; an invalid declaration
without an auditable contract raises before execution.

`candidate.delta` exposes `changed_sections`, `domain_preserved`,
`requirements_preserved`, `model_preserved`, `commitment`, `differences` and
explanatory `notes`. Differences retain before/after values and presence flags
at JSON Pointer paths in the full contract and distance declaration. Ordered
sequences are compared as whole values. A missing value remains distinct from
an explicitly declared null.

The commitment is `preserved`, `revised` or `unknown`. A verified failure followed
by a completed passing audit is `restored` only for a preserved commitment.
Other comparable transitions are `regressed`, `still_failing`, `still_passing`
and `inconclusive`. A revised or unknown commitment yields `not_comparable`.
A partial audit's verified failure witness can establish failure; unresolved
execution cannot establish restoration from a passing verdict alone.

## What preservation means

Domain preservation requires exact equality of the declared manifests. The
comparison does not infer set containment, equivalent scenario reorderings or
probabilistic relationships. A changed domain is a revised commitment.

For built-in manifests, the comparison checks selected requirement identities,
definitions, physical units, relations and tolerances, together with their
service context. Heat demand, storage initial/terminal rules and period lengths,
startup load profiles/background load/horizon, and heat-cascade stream service
are included. Changing one marks the commitment as revised.

Linear control bounds and coupled operating limits, reference equipment design,
and heat-cascade utility capacities are equipment declarations. These can change
while service is preserved. Fixed commands, startup schedules and information
permissions are operating declarations: a restoration under a new policy does
not establish that the original policy works. Storage's earlier-notification
case retains the same physical futures and explicitly changes observation
permissions.

An `AffineRequirement` is a service declaration even when the caller gives it an
equipment-like name. Changing its limit is conservatively a service revision.
Use linear control bounds/operating limits or an appropriate reference design
field to declare equipment separately from required service.

Changed physical model equations yield an unknown commitment. Unknown adapter
families retain complete structural differences, but changed declarations do
not establish service equivalence. Identical contracts can be compared. These
labels concern the declared contracts and retain the adapters' evidence scope;
for example, a passing heat-cascade comparison remains a utility-target result.

Distance settings are recorded separately. This API executes audits, so it does
not recompute failure distances or infer improvements in distance to failure.

[Frozen affine controllers](frozen-controllers.md) use the same comparison API.
A controller revision is an operating change. Its audit retains actual commands
and `fixed_policy_failure` witnesses, and can be compared with a separate
adjustable-operation benchmark without claiming the failed rule proves physical
infeasibility under other rules.

## Observed margins and portable evidence

`comparison.requirements` and `candidate.requirements` summarize selected
requirements across the stored audit responses. Each summary includes its unit,
relation, tolerance, counts by assessment, supporting evaluation references,
and the minimum `observed_margin`. Its scope is explicitly `stored_responses`.
For a signed inequality residual `r <= 0`, margin is `-r`; for equality it is
`-abs(r)`. Array coordinates are included in the minimum. Assessment counts are
per stored response, not per array coordinate. Tolerance is reported separately.

These are observed physical margins, not global worst-case bounds or guaranteed
reserve. Different audits may evaluate different points or choose different
feasible dispatches. A positive observed margin can coexist with failure at
another realization. An infeasible dispatch without residuals has no observed
margin (`None`); its joint infeasibility witness remains in the audit. No
individual violations or margin deltas are invented for such a witness.

```python
from operange import ChangeComparison

comparison = run_example()
serialized = comparison.to_json(compact=True)
loaded = ChangeComparison.from_json(serialized)
assert loaded.comparison_id == comparison.comparison_id
```

The `engineering_change_comparison/v1` envelope embeds unchanged
`process_result/v1` audits, deduplicating identical audit identities. Compact
export also deduplicates declarations inside each audit. Both encodings retain
the same content-derived comparison identity. Use `ChangeComparison.from_json`
to read this envelope; `result_from_json` continues to read individual results.
Loading checks audit identities/references and recomputes the structural
differences, transitions and summaries. It rejects inconsistent or unused
artifacts and never runs a model or solver. Preserve model implementations and
provenance separately when independent physical replay is required.
