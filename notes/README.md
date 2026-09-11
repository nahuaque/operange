# Exploring `operange`

Status: design exploration with static and finite-tree prototypes, 2026-09-11.
Origin: the `updatesupport` branch `experimental/updatesupport-process`, created from `main` at `0e52c1c`.
These design notes now accompany the standalone Operange project.

The [prototype package](../README.md) now
implements the static constant-COP heat-recovery example with SciPy, named
uncertainty coordinates, verified witnesses, repair comparison, and a portable
frozen contract. A two-period storage example adds fixed, causal, and
perfect-foresight operation, a minimal pair of conflicting futures, and
equipment/information repairs. Both now use the [shared DSL interfaces](dsl-primitives.md)
for domains, coordinates, recourse, distance and claims. The
[geometry slice](uncertainty-geometries.md) adds labeled vectors, polytopes,
simplexes, budget sets, ellipsoids and typed composition. The
[fixed affine adapter](affine-process-adapter.md) connects those support bounds
to caller-declared physical responses, sensitivities and requirement audits.
An opt-in [quadratic/residopt experiment](quadratic-residopt-experiment.md)
checks nonlinear fixed responses over ellipsoids and measures backend reuse.
The [coupled design experiment](coupled-design-experiment.md) uses those audits
to verify shared equipment sizes optimized by native, direct SDP, and residopt
backends, with independently checked objective bounds.
Broader dynamic APIs
below remain design proposals.

The target [package boundary](package-boundary.md) is a process-engineering
sensitivity and robustness DSL consumed by downstream code. It owns process
semantics and structured numerical evidence. Economics, business selection,
attribution, and presentation belong to consumers. Markdown output and CAPEX
ranking now live in repository examples outside the wheel. See the
[getting-started guide](../docs/index.md) and
[API guide](../docs/api.md). The residopt investigation
is parked while the engineering package is prepared for release.

```bash
uv run python -m examples.heat_recovery
uv run python -m examples.thermal_storage
```

Support the workflow of `updatesupport` through composable engineering results:

> State a claim, try to break it, explain the witness, compare repairs, freeze
> the selected contract, and evaluate it on later evidence.

Apply that workflow to a fixed process design, uncertain physical conditions,
and the operating adjustments available to an operator or controller. The
central question becomes:

> What is the smallest admissible disturbance that this design cannot recover
> from, and which change would make the original claim hold again?

These notes develop the supplied proposal into provisional contracts and a
small, checkable example. They are deliberately outside the published `docs/`
tree while the direction is unsettled. All proposed names and code sketches
are illustrative unless documented in the prototype package.

## Reading order

1. [Package boundary](package-boundary.md): the engineering DSL, downstream
   ownership, and the target package scope.
2. [Result contract](result-contract.md) and [worked cases](result-contract-cases.md):
   three proposed typed results, status/evidence rules, and analytical acceptance cases.
3. [Uncertainty and recourse](uncertainty-and-recourse.md): what varies, what
   operators know, what they may change, and how uncertainty sets compose;
   [implemented geometries](uncertainty-geometries.md) gives the concrete API,
   with [fixed affine claims](affine-process-adapter.md) as a worked consumer.
4. [Evidence and backends](evidence-and-backends.md): what constitutes a
   counterexample or certificate, solver boundaries, repairs, and freezing.
5. [Heat-recovery example](heat-recovery-example.md): a worked analytical
   witness and repair comparison, followed by a trajectory extension.
6. [Thermal-storage example](thermal-storage-example.md): implemented finite-tree
   control, incompatible preparation decisions, and the value of earlier information.
7. [Roadmap and open decisions](roadmap.md): implemented slices and remaining
   questions.

## What carries across

| Existing workflow | Proposed process interpretation | Important difference |
| --- | --- | --- |
| Claim and decision rule | Purity, production, energy use, or operating-limit requirement | Specify units, time aggregation, and control authority |
| Admissible composition shifts | Physical realizations, paths, faults, and model choices | A realization need not be a probability distribution |
| Minimum breaking witness | Nearest disturbance defeating the declared recourse | Failure of one attempted control is insufficient |
| Report refinement | Equipment, control, maintenance, or operating-envelope change | A lower production target changes the claim |
| Representation frontier | Engineering audit results for caller-supplied changes | Consumers value and rank candidates; evidence retains evaluation scope |
| Frozen reporting policy | Frozen design, model, uncertainty, controller, and claim contract | Causal replay matters for time-dependent operation |

The baseline [claim workflow](https://github.com/nahuaque/updatesupport/blob/08b3652bb6f8e9c7437b4329632c6e6583e9b96a/docs/reporting-claims.md),
[minimum witness](https://github.com/nahuaque/updatesupport/blob/08b3652bb6f8e9c7437b4329632c6e6583e9b96a/docs/minimum-claim-breaking-witness.md), and
[frozen policy](https://github.com/nahuaque/updatesupport/blob/08b3652bb6f8e9c7437b4329632c6e6583e9b96a/docs/frozen-public-report-policy.md) provide product
precedents. At the branch point, the minimum witness preserves public
marginals but does **not** intersect the inverse solve with the claim's forward
Q preset. A process witness must explicitly belong to its declared uncertainty
set; that behavior needs a new contract.

## Provisional choices

| Question | Working position |
| --- | --- |
| Package boundary | A process-engineering sensitivity and robustness DSL, in the standalone `operange` distribution |
| Existing core | Preserve its finite-table semantics; carry over evidence conventions where appropriate |
| First user | An engineer with an explicit process model and a concrete capacity or performance claim |
| First demonstration | Small heat-recovery model with analytical recourse, witnesses, and finite repair candidates |
| First uncertainty | Bounded continuous realizations and explicit finite scenarios |
| Trajectories | Two-period causal example implemented; general dynamic solver and frozen-controller replay remain future work |
| Solver strategy | Capabilities and verified bounds determine supported operations and guarantee strength |
| Main output | Typed process evaluations, sensitivities, robustness results, and provenance for downstream code |

Operange now has its own repository and distribution. Its responsibility
remains the engineering DSL; economic reporting belongs to downstream code.

## What would make this useful

A useful result should let a consumer reconstruct the failure: the disturbance
and its units, the controls that were allowed, the best recoverable outcome,
the binding equipment limit, and the evidence that justifies the conclusion.
It should also distinguish a repair that fixes one witness from a repair that
survives the whole declared stress set.

Rich uncertainty classes and explicit sensitivity contracts support these
results. Applications can compose them into a complete decision workflow
while the package stays focused on engineering semantics and evidence.

The most consequential open question is whether practical simulator-backed
searches can produce useful witnesses with sufficiently clear recourse
evidence. The analytical example establishes the semantics before tackling
that question.
