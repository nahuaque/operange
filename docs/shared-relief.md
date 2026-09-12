# Find one equipment-relief plan for the whole domain

`claim.audit_result(relief=...)` chooses **one shared set of operating-limit
changes** that permits dispatch at every scenario. The equipment plan is fixed
before uncertainty is revealed; permitted operating commands can vary after
observing the inputs.

The built-in `LinearProcessAdapter` supports `FiniteSet`, `BoxSet` and
`ConvexHullSet`. Boxes and hulls use exhaustive generators and the
[continuous dispatch proof](continuous-dispatch.md). A pointwise
`evaluate_result(..., relief=...)` still repairs only that realization.

## Declare the shared objective

Use the same [joint relief declarations](distance-and-relief.md): distinct
operating limits, each with a maximum relaxation, positive physical scale,
matching unit and optional positive weight.

```python
result = claim.audit_result(
    backend="scipy",
    relief={
        "changes": [
            {"constraint": "boiler_a_capacity", "maximum": 2,
             "scale": 1, "unit": "MW"},
            {"constraint": "shared_fuel", "maximum": 2,
             "scale": 1, "unit": "MW"},
            {"constraint": "boiler_a_minimum", "maximum": 3,
             "scale": 1, "unit": "MW"},
        ],
        "objective": "linear",
        "tolerance": 1e-8,
        "max_scenarios": 256,
    },
)
plan = next(e.to_dict()["details"] for e in result.evidence
            if e.evidence_id == "relief")
```

| Objective | Dimensionless expression | Backend |
| --- | --- | --- |
| `linear` | `sum(weight * relief / scale)` | SciPy or CVXPY |
| `quadratic` | `sum(weight * (relief / scale)**2)` | CVXPY |

Equipment changes count once in this objective. It does not aggregate scenario
repair objectives or infer probabilities. The model's dispatch objective is
preserved but does not select equipment. Scales and weights are caller
preferences, not inferred prices.

Upper limits relax upward; lower limits relax downward. Service requirements,
unselected operating limits, hard control bounds, fixed commands and observation
permissions remain in force. Declare contemplated capacity changes as operating
limits within a hard control box that permits the study.

## Cover both low production and peak demand

The example adds an 8 MW minimum output for boiler A. At low production, total
steam demand is only 6 MW, so the minimum must decrease. Simultaneous peaks need
20 MW of steam and require more fuel capacity or greater capacity in the more
efficient boiler A. Repairing either condition alone leaves the other problem.

```{literalinclude} ../examples/shared_relief.py
:language: python
:start-at: def example():
:end-before: if __name__
```

With zero physical tolerances and unit weights/scales:

| Objective | Boiler A capacity increase | Fuel increase | Minimum boiler A output | Objective value |
| --- | --- | --- | --- | --- |
| Linear | 0 MW | 1 MW | 6 MW, down from 8 | 3 |
| Quadratic | About 0.48 MW | About 0.64 MW | 6 MW, down from 8 | About 4.64 |

The example solves the finite modes, their continuous hull and the declared
load box. Each proposed plan receives a passing audit over its original domain.

```bash
uv run python -m examples.shared_relief
uv run --extra cvxpy python -m examples.shared_relief --objective quadratic --backend cvxpy
```

## Read the plan and its re-audit

`result.payload.verdict` concerns the **original equipment**. Finding a plan
does not turn an originally failing verdict into a pass. The `relief` evidence
reports one of these resolutions:

- `minimum_verified`: a checked full-domain candidate and global lower bound
  close within the objective tolerance.
- `bounded`: a checked full-domain candidate with an open optimality gap.
- `unreachable`: an exact contradiction proves the allowed changes cannot
  restore the whole domain.
- `unresolved`: no checked full-domain candidate or contradiction was obtained.
  A valid lower bound may still be available.

An open relief gap makes overall execution `unresolved` while preserving the
original physical verdict. An already passing baseline returns a checked
zero-relief plan. Unresolved baseline evaluations are retained: their constraints
still enter the shared problem and are never omitted as passing scenarios.

When a candidate exists, inspect `candidate.changes`, `candidate.dispatches`
and the separately checked audit:

```python
import json
from operange import result_from_json

if plan.get("candidate") is not None:
    repaired = result_from_json(json.dumps(plan["candidate"]["reaudit"]))
    print(repaired.payload.verdict, repaired.payload.coverage.method)
```

This re-audit recompiles the changed physical model and checks the recorded
dispatches, retaining the original permissions and checking membership again.
It covers every original scenario or generator without asking another optimizer
to invent replacement commands. Any dispatch-objective bounds concern those
recorded commands; optimal dispatch is not required for restoration.

The re-audit is embedded as a compact `process_result_bundle/v1` with its own
contract and identity. The outer result supports both existing export formats.
Loading checks records and identities without re-proving numerical evidence;
load the nested audit explicitly to validate its result structure.

## Formulation and limits

One bounded dispatch block per distinct physical realization is coupled to one
shared relief vector. Every original row is retained; only selected operating
limits receive relief coefficients. Duplicate realizations may share a dispatch
block, while every scenario label remains in the coverage record.

The existing bounded linear/quadratic solver proposes commands and multipliers.
An exact separable Lagrangian check over the full program supplies a lower bound.
Evidence exports the matrix, right-hand sides, objective coefficients,
row-to-scenario mapping, control transformations and certificate. Unreachability
requires a positive exact contradiction gap; solver status alone is insufficient.

Candidate limits round outward in their permitted direction. Actual physical
changes must remain within the caller's maxima. Only a passing full-domain
replay supplies an objective upper bound. Near-bound commands may be snapped
and repaired as candidates, but every returned command passes exact checks.
Numerical tolerances never relax the physical contract.

A common relief vector with feasible dispatch at every box/hull generator
establishes feasibility throughout the domain by convex interpolation. Thus the
finite program solves the continuous equipment problem under this fixed linear
model. It does not synthesize a controller or certify rounding of interpolated
commands.

`relief.max_scenarios` defaults to 256 and counts all scenarios or generators,
including duplicate labels. Continuous audits also retain `max_vertices`.
Exceeding either budget rejects the query before solving; increase the relevant
limit explicitly to permit full enumeration. The dense program grows with the
total scenario controls and constraints, so large studies may need a smaller
scope or a future decomposition method.

Only fixed or fully observed single-stage operation is supported. Partial
observation, integer equipment choices, uncertain control coefficients, general
polytopes, other domain compositions, saved-controller relief and multistage
storage planning remain outside this operator.
