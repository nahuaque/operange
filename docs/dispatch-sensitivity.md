# Dispatch and bottleneck sensitivities

`claim.dispatch_sensitivity()` explains how the **optimized objective value**
changes when one input, operating limit or control bound changes. It reports
checked left and right slopes, units, and an interval over which each slope is
valid. This answers questions such as fuel needed for extra steam demand or
fuel saved by increasing efficient-boiler capacity.

```python
from operange import DispatchChange

result = claim.dispatch_sensitivity(
    {"load": 16},
    changes=(
        DispatchChange("input", "load", "MW", step=2),
        DispatchChange("control_upper", "a", "MW", step=2),
        DispatchChange("operating_limit", "fuel_supply", "MW"),
    ),
)
for item in result.sensitivities:
    print(item.change.change_id, item.resolution)
    print(item.backward.value, item.forward.value, item.forward.unit)
    print(item.forward.valid_step, item.forward.supporting_constraints)
```

The model must be a declarative `LinearProcessAdapter` with a `LinearObjective`.
The default backend is SciPy; `backend="cvxpy"` selects the optional CVXPY
proposal route. Both use the same independent exact checks. Compiled
[equipment networks](equipment-networks.md) work through this API too.

## Read the engineering results

The [complete example](https://github.com/nahuaque/operange/blob/main/examples/dispatch_sensitivity.py)
has boiler A delivering up to 12 MW at 1.25 MW fuel per MW steam, and boiler B
delivering up to 10 MW at 2 MW fuel per MW steam. Shared fuel capacity is 32 MW.

| Study | Backward slope | Forward slope | Interpretation |
| --- | --- | --- | --- |
| Load at 16 MW | 2 | 2 | An extra MW of demand costs 2 MW of fuel. |
| Boiler A capacity at 12 MW, load at 16 MW | -0.75 | -0.75 | An extra MW of efficient capacity saves 0.75 MW of fuel. |
| Shared fuel capacity at 32 MW, load at 16 MW | 0 | 0 | More fuel availability does not reduce fuel consumption here. |
| Load at 12 MW | 1.25 | 2 | A kink: additional load starts using the less efficient boiler. |
| Boiler A upgrade with a separate 12 MW burner limit | -0.75 | 0 | Reducing capacity hurts; increasing it cannot help until the burner limit also changes. |
| Load at 20.5 MW | 2 | Infeasible | Any further load increase exceeds the shared fuel supply. |

**Both slopes use the convention of increasing the coordinate.** A backward
slope of 2 means that decreasing load by 0.1 MW decreases fuel use by 0.2 MW.
The negative capacity slope represents a saving from an increase in capacity.

`valid_step` is a nonnegative distance from the base, in the change's unit:
the backward interval is `[-valid_step, 0]` and the forward interval is
`[0, valid_step]`. It never exceeds the requested `step`. For example, at 16 MW
load and `step=100`, the example's certified constant-slope intervals extend
4 MW backward to the boiler threshold and 4.5 MW forward to the fuel boundary.
The interval can be conservative; it is not promised to be the largest possible.

The example uses zero balance tolerances to place its thresholds exactly at
12 and 20.5 MW. In general, sensitivities include every declared residual
tolerance, so thresholds and slopes apply to that precise mathematical model.

## Choose the changes

`DispatchChange(kind, name, unit, step=1)` varies one numeric declaration while
holding all others fixed. Units must exactly match the target; no conversion
or normalization is inferred.

| Kind | Target | Positive direction |
| --- | --- | --- |
| `input` | A physical input coordinate | Increase its realization value. |
| `operating_limit` | An `AffineRequirement` in `model.operating_limits` | Increase its numeric `limit`. For a `ge` limit this tightens the minimum. |
| `control_upper` | A declared control | Increase its upper bound. |
| `control_lower` | A declared control | Increase its lower bound. |
| `state_upper` | A declared algebraic state | Increase its upper bound. |
| `state_lower` | A declared algebraic state | Increase its lower bound. |

Service requirements cannot be changed with `operating_limit`. They remain
selected by the claim, and every operating limit remains present. Fixed
operating permissions remain fixed even when a bound changes. Equal lower and
upper equipment bounds are kept as separate constraints: explicitly changing
one can create room for adjustment, unless a permission fixes the command too.

Each requested change is analyzed independently. Joint upgrades, changes to
coefficients or efficiencies, quadratic objectives and integer decisions are
outside this first scope. A compiled stream capacity uses its declared variable
role: `DispatchChange("state_upper", "steam_a", "MW")` for a steam state, or
`control_upper` when that stream is declared as an actuator. Legacy network
compilation without explicit `actuators` retains control bounds for every stream.

These are **ambient** derivatives. The base must be a checked member of the
claim's domain. Nearby changes are checked against the physical equations and
operating permissions, but are not constrained to remain inside the original
uncertainty set. Thus a finite scenario study can have an ambient local
sensitivity, and a box-edge derivative may concern a point beyond that box.
Use an audit or an [operating-limit query](operating-limits.md) for guarantees
over an uncertainty envelope.

## Distinguish the value function from a controller

The response is the best attainable value of the declared linear objective,
under fixed or fully observed single-stage operation. It is not the derivative
of one numerically selected command vector. Several dispatches can share the
same optimal value and still have a well-defined objective slope.

Saved controllers and partial-observation adjustable policies are unsupported;
the query does not redispatch their commands or broaden their observations.
Measurement-based operation continues to require an explicit controller. The
existing `sensitivity_result()` API retains its fixed-response derivative
contract. This separate result supports capacity changes, which are not input
coordinates in that contract.

`supporting_constraints` lists rows with positive multipliers in the particular
certificate. It helps explain the slope: the blocked-upgrade example's forward
certificate names `operating_limit:burner_limit`. This list need not be unique
at degeneracy and is not a minimal conflict or a unique ranking of equipment.
Changing several of those constraints at once requires a new study.

## What proves a slope

After substituting the base inputs, a change of size `delta` produces the LP:

```text
minimize    k + delta*g + c*x
subject to  A*x <= b + delta*h
```

Maximization is converted by negating the objective. Hard control bounds and
fixed commands appear as explicit rows, so changing a bound does not silently
change the coordinate system or the operating permissions.

The numerical solver proposes a basis. Exact rational arithmetic on the
declared floats reconstructs a real-valued optimal dispatch `x0`, a feasible
affine path `x0 + delta*dx`, and nonnegative multipliers `lambda`. The checker
requires:

```text
c + A.T*lambda = 0
k + c*x0 = k - lambda*b
g + c*dx = g - lambda*h
```

It also checks every physical inequality at both ends of the reported interval.
Because the path and right-hand sides are affine, endpoint feasibility covers
the whole interval. Matching primal and dual values proves that the optimum is
exactly affine there, with slope `g + c*dx`. A numerical finite difference or an
approximately optimal multiplier alone does not establish a derivative.

For an infeasible direction, a separate nonnegative ray must satisfy
`A.T*lambda = 0`, `lambda*b = 0`, and a strictly negative directional change in
`lambda*h`. That proves that every positive step in the reported direction is
infeasible, rather than merely observing a failed solve at one nearby point.

The search first tries a base basis, then probes at the requested step and
repeatedly halves it. `max_refinements=16` caps new proposals per side; values
from 0 through 32 are supported. Up to eight exact basis exchanges can repair
numerically rounded active sets. This is bounded candidate recovery, not a
complete exact LP solver. Failure to recover a checked basis remains unresolved.
The initial size limits are 64 controls, 64 changes and a conservative bound of
4096 physical rows. There is no corner enumeration.

## Inspect and save the report

`DispatchSensitivityResult` retains the base evaluation, source contract,
requested changes, numerical attempts and the checked directional reports:

| Field | Meaning |
| --- | --- |
| `execution == "completed"` | Every requested direction has a slope or an exclusion certificate. |
| `execution == "unresolved"` | Some directions remain unresolved, or no feasible base was checked. Any independently proved sides remain available. |
| `execution == "unsupported"` / `"invalid"` | Unsupported binding or invalid request; no slopes are asserted. |
| `item.resolution` | `differentiable`, `kink`, `one_sided`, `isolated_feasible_point`, or `unresolved`. |
| `side.status` | `verified`, `infeasible`, or `unresolved`. Only `verified` carries a derivative. |
| `side.slope_exact` | Exact rational derivative as a string. |
| `side.value`, `side.lower`, `side.upper` | Rounded display value and outward-rounded bounds on the exact derivative. |
| `side.base_value` | Rounded optimum from the local certificate. It can differ slightly from the base evaluator's floating-point candidate value. |
| `side.proof` | Physical rational controls and rates, row identities, multipliers and base value. |

The certificate concerns real-valued optimization. Its rational controls are
witnesses for the mathematical value function; they are not a deployable
controller or a claim that rounded commands satisfy every row. The retained base
evaluation separately checks the actual floating-point dispatch it returned.

```python
from operange import DispatchSensitivityResult, verify_result

saved = result.to_json(compact=True)
restored = DispatchSensitivityResult.from_json(saved)
report = verify_result(saved, expected_contract_id=claim.contract.ref.artifact_id)
```

The `dispatch_sensitivity/v1` loader checks structure and content identity.
`verify_result` independently rebuilds physical rows and rechecks the primal,
dual, interval and exclusion mathematics, without calling the producer's
compiler, solver or model query. It supports the same box, finite-set and
explicit-hull-generator base membership as the other linear certificate checks.
Verification can accept a partially resolved report because it checks only the
conclusions actually asserted; it does not turn unresolved sides into results.

Run the complete example or verify a saved report:

```bash
uv run python -m examples.dispatch_sensitivity
python -m operange.verify sensitivity.json
```
