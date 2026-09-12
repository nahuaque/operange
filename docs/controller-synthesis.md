# Synthesize an affine controller

`claim.synthesize_controller()` chooses one affine command rule per control,
using only its permitted observations. It then audits the saved float
coefficients and actual rounded commands over the original domain. A successful
search returns a controller ready for the existing `FrozenController` format.

Install the optional backend with `pip install "operange[cvxpy]"`. The first
implementation supports `LinearProcessAdapter` with a built-in `BoxSet`,
`FiniteSet`, or explicit `ConvexHullSet`. It uses every box corner, finite case,
or hull generator. `max_vertices=256` limits the total number of generators
(including duplicate points with distinct labels); exceeding it returns
`unsupported` before solving. There is no sampling fallback.

## Choose the operating objective

```python
from operange import LinearObjective, FrozenController

search = claim.synthesize_controller(
    name="Minimum worst-case fuel",
    objective=LinearObjective("fuel"),
    max_vertices=256,
)
if search.verified:
    controller = search.controller
    audit = search.audit
    frozen = search.freeze()
    loaded = FrozenController.from_json(frozen.to_json())
    replay = loaded.audit_result()
```

Omitting `objective` uses the model's declared operating objective. A search
needs either a `LinearObjective` or a `ControlTrackingObjective`; declaring an
objective on the query preserves the original model and its identity.

| Declaration | Whole-domain synthesis objective |
| --- | --- |
| `LinearObjective("fuel")` | Minimize the largest fuel output. |
| `LinearObjective("margin", sense="maximize")` | Maximize the smallest margin output. The caller declares the affine margin as a model output. |
| `ControlTrackingObjective(targets)` | Minimize the largest sum of weighted squared, scaled departures from the target commands. |

These objectives apply across all generators, without scenario probabilities.
Worst-case tracking gives each scenario a sum of target errors and minimizes
the largest such sum. It does not minimize the sum across scenarios.

For the synthetic boiler example, dryer demand spans 8–12 MW and evaporator
demand spans 4–8 MW. Boiler A has a 12 MW steam capacity and 80% efficiency;
boiler B has a 10 MW capacity and 50% efficiency. They share a 32 MW fuel supply.
Minimum worst-case fuel synthesis finds a simple feasible rule: hold A at
12 MW and command B to the total demand minus 12 MW. Its worst-case fuel use
is enclosed at approximately 31 MW, including command rounding. The optimum
need not select a unique controller, and interior dispatch need not minimize
fuel separately at every input.

The [complete consumer example](https://github.com/nahuaque/operange/blob/main/examples/controller_synthesis.py)
also synthesizes a controller tracking preferred commands of 10 MW and 6 MW,
exports both reports, and replays a frozen controller at an interior input:

```bash
uv run --extra cvxpy python -m examples.controller_synthesis
```

## Preserve observations and physical limits

All selected service requirements, every operating limit, and every hard control
bound constrain the search and subsequent audit. Fixed permissions stay fixed.
A control with no observations can acquire only a constant command. A control
may use a subset of the inputs; those permissions are preserved even when the
separate adjustable audit cannot support that observation structure. Constant
input coordinates can be absorbed into the rule's offset.

Only the single `operation` stage is supported. Rules have no clipping,
saturation, state, or hidden observations. Calling synthesis on an already
bound controller explicitly searches for a replacement within the same
permissions. It does not modify the original controller or claim.

The numerical program uses input ranges and control spans for conditioning.
It first reserves half of each declared residual tolerance for coefficient and
command rounding, then retries with the full tolerance if no candidate passes.
Zero tolerances and hard bounds are never loosened. Each proposal is tried with
nearby simple rational coefficients and its original numerical coefficients;
both are converted into physical float coefficients and independently audited.
These are bounded candidate-generation heuristics, not completeness guarantees.

For continuous domains, passing generator executions alone does not accept a
controller. The existing continuous audit encloses its rounded commands between
generators. Boxes and explicit hulls use exact affine extrema; tracking bounds
use a convex enclosure of squared target errors plus command-rounding error.
Finite-domain acceptance covers only the declared cases.

## Read and export the result

`ControllerSynthesis` is a policy-search report, separate from a robustness
verdict about adjustable operation:

| Field | Meaning |
| --- | --- |
| `execution == "completed"`, `verified == True` | A candidate passed its physical audit and has a checked whole-domain performance enclosure. |
| `execution == "unresolved"` | The search found no candidate meeting both checks. Numerical infeasibility does not prove that no affine controller or adjustable response exists. |
| `execution == "unsupported"` | Missing CVXPY, an unsupported model/domain, or a generator count above the explicit limit. |
| `execution == "invalid"` | An invalid objective, permission declaration, name, or limit. |
| `controller`, `audit` | The saved candidate and its independent audit, when available. An unresolved search can retain a failing candidate for diagnosis. |
| `claim` | The candidate bound to the original domain, model, requirements and permissions. Raises if no candidate exists. |
| `evidence` | Generator coverage, numerical solver attempts, candidate audit references, and checked objective bounds. |

The `objective` evidence contains outward-rounded `lower` and `upper` bounds on
the **worst-case performance of this controller**. `guaranteed_value` is the
upper bound for minimization and the lower bound for maximization. Numerical
solver status and cost are retained separately. Optimality across controllers is
`not_certified`; the objective declaration's tolerance does not imply a verified
optimality gap. A failed candidate audit concerns only that prescribed rule.

`search.to_json(compact=True)` exports the `controller_synthesis/v1` report,
including the source contract, query objective, candidate manifest, evidence,
and standard audit result. The report is an export for inspection; replay uses
`AffineController.from_json`, `result_from_json`, or the frozen artifact.
`search.freeze()` requires verified synthesis. It exports the existing
`frozen_controller/v1` artifact without adding solver runtime dependencies to
replay. The general `claim.freeze()` API remains available for reproducible
failed-controller counterexamples.

Use `frozen.as_claim(new_domain).audit_result()` to assess new operating support.
The original synthesis and frozen artifact retain their original scope.
