# Thermal storage and the timing of information

Implemented in the [prototype package](../README.md).
All data and costs below are synthetic. This example tests whether causal
operation changes the claim, witness, and repair workflow in a useful way.
Its report and CAPEX ranking are prototype conveniences. Under the target
[package boundary](package-boundary.md), the core supplies engineering results
and consumers own valuation, selection, and presentation.

## The claim

Meet heat demand and absorb all mandatory recovered heat over a declared
two-period scenario tree. The store starts empty. One hour of preparation is
followed by a one-hour event. Initial equipment has:

| Quantity | Value |
| --- | --- |
| Storage capacity | 1 MWh thermal |
| Charging and discharging power | 1 MW thermal each |
| Charge and discharge efficiency | 1 |
| Standing loss | 0 |
| Final energy | Between zero and installed capacity |

During preparation, an external heat source can charge the store. During the
event, heat must balance exactly: recovered heat and storage discharge supply
demand and storage charging. There is no external event supply, discarded heat,
unmet demand, or simultaneous charging and discharging. Preparation cannot
discharge. These permissions are part of the claim.

There are two possible events, with the same observation before preparation:

| Future | Demand (MW) | Mandatory recovered heat (MW) | Required prepared energy (MWh) |
| --- | --- | --- | --- |
| Demand surge | 1 | 0 | Exactly 1 |
| Heat surplus | 0 | 1 | Exactly 0 |

The exact event becomes observable after preparation, before event actions.
The surplus future needs empty storage because it must absorb the incoming
heat. The demand future needs full storage because no event supply is available.

## Three operating contracts

| Contract | Preparation permissions | Baseline result |
| --- | --- | --- |
| Fixed rule | Charge at a supplied constant, here 0.5 MW | Fails; one future refutes this rule |
| Causal policy class | Choose one action for each declared preparation observation | Fails; the two futures require incompatible shared actions |
| Perfect foresight | Choose preparation separately for each future | Passes; prepare 1 MWh for demand and 0 MWh for surplus |

The causal LP creates one preparation variable and one prepared-state variable
per observation label. Every future carrying that label uses those same
variables. Distinct scenario names do not reveal the future. Labels can also
represent partial information: a signal may distinguish some futures while
leaving others indistinguishable.

The fixed rule ignores preparation observations. Event actions in all three
contracts are determined by the revealed demand and recovered heat through
exact balance and the prohibition on simultaneous operation. No unspecified
event optimization makes the fixed rule more permissive.

The causal witness is a **minimum-cardinality pair of futures**, not a smallest
disturbance or a probability of failure. Each future alone admits an operating
plan; together they require prepared energy at least 1 MWh and at most 0 MWh.
The reported 1 MWh gap is the separation of incompatible state requirements.
In general, a single individually infeasible future yields a one-future witness.

## Physical equations and independent check

Let `dt0` and `dt1` be the preparation and event durations in hours, `K` the
capacity in MWh, `C`/`D` the charging/discharging power limits in MW, and `E0`
the fixed initial energy. With efficiencies `eta_c`, `eta_d` and standing-loss
rate `lambda` per hour, define:

```text
alpha = 1 - lambda * dt0
beta  = 1 - lambda * dt1

E1 = alpha * E0 + eta_c * c0 * dt0
c - b = recovered_heat - demand
E2 = beta * E1 + (eta_c * c - b / eta_d) * dt1

0 <= c0 <= C
0 <= c <= C, 0 <= b <= D
0 <= E1, E2 <= K
terminal_min <= E2 <= terminal_max, when terminal_max is supplied
```

This is a discrete model with losses applied to the opening state of each
period. It requires positive durations, efficiencies in `(0, 1]`, and
`lambda * dt < 1` in both periods. It does not claim continuous-time
thermodynamic fidelity. Initial energy must lie within installed capacity.
An impossible terminal requirement is a valid failing claim.

Exact balance and the allowed operating mode determine the event controls:

```text
q = recovered_heat - demand
c = max(q, 0)
b = max(-q, 0)
offset = (eta_c * c - b / eta_d) * dt1
U = min(K, terminal_max) if terminal_max is supplied, otherwise K

reachable_low  = alpha * E0
reachable_high = min(K, reachable_low + eta_c * C * dt0)
L = max(reachable_low, (terminal_min - offset) / beta)
H = min(reachable_high, (U - offset) / beta)
```

The analytical reference checks event power limits and derives the complete
allowable preparation interval `[L, H]` for each future independently of the LP
matrices. A causal group is feasible precisely when its event power limits
hold and its intervals intersect. Disjoint intervals in one dimension have a
pair whose lower and upper bounds conflict, which makes the witness checkable
without a general combinatorial search.

The LP minimizes the sum of preparation power over observation groups to
select a reproducible feasible plan. This is not an expected operating cost:
there are no scenario probabilities. The reference calculation also checks
this optimum. The shared SciPy LP interface recomputes primal residuals.
Both checks must agree before the audit returns `pass` or `fail`; solver
errors, unresolved termination, and disagreement produce `inconclusive`.
The default tolerance is `1e-8`, applied to the model's MW, MWh, and objective
residuals. Results are numerical evidence for the declared model.

## Repairs preserve the service requirement

| Candidate | Synthetic added CAPEX | Causal result | Explanation |
| --- | --- | --- | --- |
| Baseline | £0 | Fail | Incompatible preparation states |
| Increase charging power to 2 MW | £10,000 | Fail | Does not create storage headroom |
| Reveal exact event identity before preparation | £15,000 | Pass | Preparation can depend on the observed event |
| Increase capacity to 2 MWh | £40,000 | Pass | Prepare 1 MWh for either event, ending at 0 or 2 MWh |

All four designs pass the perfect-foresight benchmark. The example therefore
shows a decision the static or perfect-foresight view would miss: baseline
equipment needs a repair under the original information timing.

The information intervention represents an ideal upstream notification, such
as a scheduled event whose identity can be communicated earlier. It explicitly
assumes a correct early signal. An ordinary imperfect forecast needs a tree
that retains every compatible future under each signal. The £15,000 cost is
an illustrative input, not an estimate of real instrumentation or forecasting.

Every repair retains the physical futures, durations, initial state, and
terminal service requirements. An explicit terminal maximum stays fixed when
equipment grows; the default maximum follows installed capacity because it is
an equipment limit. The comparison picks the cheapest verified causal pass in
the supplied catalog. Unresolved candidates remain visible and may change
that selection. It does not optimize beyond the listed candidates.

## Run and inspect

From the repository root:

```bash
uv run --locked python -m examples.thermal_storage \
  --output data/process/storage.md \
  --json-output data/process/storage.json
```

The Markdown report shows all three contracts, their witnesses or plans, and
the repair comparison. JSON includes every audit, observation label, physical
input, selected action, solver version, tolerance, and numerical evidence.

```python
from operange.reference import (
    ScenarioTree, StorageDesign, StorageRequirement, StorageScenario,
    ThermalStorageClaim,
)

claim = ThermalStorageClaim(
    name="Meet heat demand and absorb mandatory recovered heat",
    design=StorageDesign(capacity_mwh=1, charging_power_mw=1, discharging_power_mw=1),
    requirement=StorageRequirement(initial_energy_mwh=0, terminal_min_mwh=0),
    tree=ScenarioTree((
        StorageScenario("Demand surge", 1, 0, "Synthetic demand event"),
        StorageScenario("Heat surplus", 0, 1, "Synthetic recovery event"),
    ), preparation_hours=1, event_hours=1),
)
print(claim.audit_result("fixed", fixed_preparation_power_mw=0.5).to_json())
print(claim.audit_result("causal").to_json())
print(claim.audit_result("perfect_foresight").to_json())
```

## Scope and next evidence

Coverage is the entire declared **finite two-period tree** and the permitted
policy class. It does not extend to unlisted trajectories, intermediate
observations, continuous uncertainty sets, or an indefinite sequence of events.
The terminal condition is deliberately a finite-horizon requirement, not a
cyclic reset. The exported plans are review evidence; they are not a portable
frozen storage controller or a historical backtest.

The next extension should freeze a causal controller and replay ordered
held-out events while preserving observation timing. That would test whether
the finite-tree reasoning remains useful when a controller faces later
evidence. A general dynamic optimizer or simulator integration can follow
when that exercise requires it.
