# Worked cases for the engineering result contract

Acceptance references for the [result contract](result-contract.md), 2026-09-11.
The [contract tests](../tests/test_result_contract.py)
exercise the analytical numbers, first derivatives, failure semantics, and
portable exports. Payload illustrations below omit common envelope fields and
are explanatory rather than literal JSON. Fixed-control derivatives and broader
dynamic sensitivity operators remain unsupported in this implementation.

## 1. Heat recovery: feasibility, service, and optimum

Use the existing constant-COP model: source capacity 6 MW thermal, compressor
capacity 2 MW electrical, COP 4, base heat demand 6 MW, and zero reserve.
`a` is source derating and `b` is demand increase, both expressed as fractions
in `[0, 0.25]`.

```text
maximum delivered heat q(a) = min(8, 8*(1-a)) MW thermal
required heat h(b) = 6*(1+b) MW thermal
delivery margin m = q - h
requirement residual r = h - q = -m
```

At `a = 0.10`, `b = 0.05`, optimization of delivered heat gives:

| Quantity/assessment | Value |
| --- | --- |
| Domain membership | Inside |
| Physical feasibility | Feasible |
| Delivered heat | 7.2 MW thermal |
| Electrical input | 1.8 MW electrical |
| Used source heat | 5.4 MW thermal |
| Required heat | 6.3 MW thermal |
| Delivery margin | +0.9 MW thermal |
| Requirement residual | −0.9 MW thermal; satisfied |
| Maximum-delivery bounds | Lower = upper = 7.2 MW, with LP and independent capacity evidence |

At the box's upper corner, `a = b = 0.25`, the optimum is 6 MW against
7.5 MW required. The physical model still admits operation. Thus the evaluation
is **feasible**, its requirement is **violated**, and the separate robustness
audit is **fail**. The optimum upper bound establishes that another allowed
static control cannot supply the missing heat. Checking one arbitrary control
would not establish that last conclusion.

Migration must therefore split today's `OperationResult.status`, which says
`pass`/`fail` for delivered heat, into physical feasibility and requirement
assessment. It must also preserve the independent optimum check currently
performed by the heat adapter.

## 2. Sensitivity under a declared operating response

At the first point above, source capacity limits the optimal delivered heat.
With maximum-delivery recourse, the local Jacobian is:

| Output | With respect to `a` | With respect to `b` |
| --- | ---: | ---: |
| Maximum delivered heat | −8 MW per unit fraction | 0 |
| Required heat | 0 | +6 MW per unit fraction |
| Delivery margin | −8 MW per unit fraction | −6 MW per unit fraction |
| Requirement residual | +8 MW per unit fraction | +6 MW per unit fraction |

For normalized coordinates `za = a/0.25`, `zb = b/0.25`, the margin Jacobian
is `[-2, -1.5]` MW per normalized coordinate. It is not `[-8, -6]` in those
coordinates. An illustrative derivative payload is:

```json
{
  "operator": "jacobian",
  "output_coordinates": ["heat.delivery_margin"],
  "input_coordinates": ["source_derating", "demand_increase"],
  "coordinate_space": "physical",
  "values": [[-8.0, -6.0]]
}
```

The contract registry supplies each coordinate's unit, axes, and definition;
the surrounding sensitivity result supplies the base evaluation, method, and
response contract. This fragment is not a complete serializable result.

For the dimensionless path parameter `t` and direction `v = (0.01, 0)`, the
directional response of margin is −0.08 MW per unit `t`. A one-percentage-point
derating is a fraction change of `0.01`, not `1`. Here the response is exactly
linear while the same capacity branch remains active. That local statement is
not a full-domain robustness certificate.

Controls matter. Hold the original electrical input at 1.8 MW and increase
`a` above 0.10: the required source input exceeds available source heat, so the
fixed-control physical point becomes infeasible. The optimized-response slope
above must not be returned as a two-sided fixed-control derivative.

For a separate design with compressor capacity **1.8 MW**, the maximum-delivery
function has a kink at `a = 0.10`. Its derivative with respect to increasing
`a` is −8 MW per fraction; from decreasing `a` it is 0. A requested two-sided
derivative has `availability = undefined` with a nondifferentiability reason.
Explicit one-sided requests may return those slopes. This changed design has
a different contract identity from the smooth case.

These derivatives are analytical design references; the current package does
not expose a general sensitivity method.

## 3. Boundary, breaking witness, and minimum evidence

Use normalized L-infinity severity from the nominal point, with both fraction
scales equal to `0.25`. The existing analytical reference cases distinguish:

| Requested calculation | Expected result |
| --- | --- |
| Full-box audit | Fail, supported by the upper-corner optimum and monotonicity |
| First nonpositive delivery margin from the passing nominal point | Minimum severity `4/7`; `a = b = 1/7`; margin zero |
| A shortfall of at least 0.01 MW | Minimum severity `201/350`; `a = b = 201/1400`; margin −0.01 MW |

The boundary calculation establishes its search target and minimum, while the
boundary point satisfies the inclusive requirement. It does not create a
strictly breaking witness or independently assert a full-domain pass.

The 0.01 MW witness requires both membership and evidence that even maximum
delivery misses the requirement. Its minimum severity requires exclusion of
closer target-reaching realizations across all exhaustive failure branches.

If that witness is verified but one outer branch is unresolved, the new
contract should expose both facts: `verdict = fail` for the original claim,
and an unproved search minimum. It can retain the witness severity as a valid
upper bound. This is more informative than interpreting a single
`status = inconclusive` as either “no failure established” or “no minimum
established.” It must not copy a candidate distance into a lower bound.

If a larger requested shortfall is proven unreachable, the result excludes
that target only. Smaller violations may still exist. Nominal violation of the
requested breaking target yields zero severity when the nominal point is
admissible; it still needs membership and violation evidence.

## 4. Storage: a causal conflict is not an impossible path

Use the existing empty 1 MWh store, one preparation hour and one event hour,
unit efficiencies, zero loss, and 1 MW charge/discharge limits.

The demand future needs prepared energy `E1 >= 1 MWh`. The mandatory-surplus
future needs `E1 <= 0 MWh`. Both carry the same preparation observation.

| Operating contract | Robustness verdict | Required evidence |
| --- | --- | --- |
| Fixed 0.5 MW preparation | Fail | One admissible future violates the fixed rule |
| Causal policy class | Fail | Two admissible futures impose incompatible bounds on the same shared state/action |
| Perfect foresight | Pass | A feasible plan for each future, under the explicitly different information contract |

The causal witness references both futures, their common information node,
the conflicting state inequalities, and the infeasibility evidence. Each
singleton subtree is feasible. That establishes minimum cardinality two for
this conflict; it does not establish a continuous disturbance severity.

No feasible full-tree causal policy is emitted for the failed contract.
Per-future perfect-foresight solutions belong to their own result/contract and
cannot be joined into an allegedly causal policy. The coverage record identifies
the whole declared finite tree, its two leaves, and its observation structure.
There is no inferred probability or annual aggregation.

An equipment change to a 2 MWh store admits one shared preparation state of
1 MWh; the two events finish at 0 and 2 MWh. The caller can join the original
and changed engineering results. It checks that the service and external tree
were preserved, while the design digest changed. The default upper state limit
follows installed capacity; an explicitly fixed terminal maximum must retain
its value. An information change instead changes the operating/observation
contract, even if the physical futures stay identical.

## 5. Failure and partial-evidence cases

| Situation | Execution / semantic result |
| --- | --- |
| Invalid units, duplicate coordinates, or empty audit domain | `invalid`; no assessed claim or fabricated values |
| Backend lacks the requested derivative | `unsupported`; sensitivity `not_evaluated` |
| Solver timeout with no verified point | `unresolved`; feasibility `unknown` |
| Timeout with an independently checked feasible point | `unresolved`; feasibility can be `feasible`, while optimality remains unknown |
| An arbitrary feasible control misses demand | Requirement violated at that control; adjustable robustness remains unestablished |
| Complete finite audit with valid recourse evidence | Pass/fail for precisely that finite contract |
| Completed sampling, all samples pass | Full-domain verdict `inconclusive`, with partial coverage |
| Verified breaking witness, unresolved minimum search | Full-domain verdict `fail`; keep candidate upper bound and unresolved minimum |
| Conflicting solver and analytical results | Unresolved affected conclusion; preserve both checks and the diagnostic |
| Complete local derivative check | Sensitivity available at its declared scope; no robustness promotion |

These are acceptance criteria for a later implementation. Some are supported
by today's prototype checks; others, including retaining feasible points from
timed-out optimizations, require additional backend support. No missing evidence
should be manufactured to fill the proposed fields.

## Mapping the current prototype

| Current surface | Intended treatment |
| --- | --- |
| `OperationResult` | Map physical values and checks into `EvaluationResult`; split feasibility from service and preserve optimum evidence |
| `ClaimAudit` / `StorageAudit` | Map claim/recourse/coverage into `RobustnessResult`; reference evaluation artifacts |
| `ThresholdResult` | Separate claim failure from target-search resolution, bounds, and attainment |
| `StorageWitness` | Preserve information-node conflict and cardinality evidence as a witness variant |
| `SolveEvidence` | Preserve solver output and independent checks as separately scoped evidence records |
| Frozen heat-recovery contract | Retain its current schema/fingerprint; explicitly version any future conversion |
| General sensitivity | New capability, starting with the analytical heat reference and explicit unsupported cases |
| Repair comparison and rendering helpers | Keep engineering re-audit in the core; migrate selection and presentation into example/consumer code later |

No result conversion should infer a solver capability or stronger guarantee
from a legacy field name. The first later implementation can introduce typed
results alongside the examples, then adapt consumer demos without changing the
analytical acceptance values. This iteration changes documentation only.
