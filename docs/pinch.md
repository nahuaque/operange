# Heat-integration target robustness

The heat-cascade reference adapter asks whether the **minimum thermodynamic
heating and cooling requirements** fit within declared utility capacities across
a finite set of steady operating conditions. It returns targets, heat-cascade
data, pinch locations and replayable failure evidence through `process_result/v1`.

A passing target audit does **not** establish that an installed exchanger
network can meet those requirements. Each scenario permits unrestricted heat
matches and stream splitting. Actual connections, exchanger areas, fouling,
bypasses, utility temperatures and control response need a network model.

## Declare streams and operating cases

Import `SensibleHeatStream` and `HeatCascadeAdapter` from
`operange.reference`. They do not add a new uncertainty primitive
or any runtime dependency.

```{literalinclude} ../examples/pinch.py
:language: python
:start-at: from operange import
:end-before: def run_example():
```

Each stream declares its hot/cold direction, nominal supply and target
temperatures in **degC**, constant heat-capacity flow in **kW/K**, and provenance.
These quantities become named scenario inputs, for example
`hot_product:supply_temperature_c`. Heat-capacity flow means mass flow times
specific heat capacity, not specific heat capacity alone. The adapter performs
no property estimation or unit conversion. A scenario supplies all coordinates;
merge changes into `model.input_space.nominal` when constructing cases.

Hot streams cool; cold streams heat. Zero heat-capacity flow or equal endpoint
temperatures mean zero duty and are omitted from the cascade. An isothermal
phase change cannot be represented by equal endpoints in this sensible-heat
model. Negative heat-capacity flow, temperatures below absolute zero and
reversed stream directions are invalid. The caller remains responsible for
ensuring that supplied stream scenarios correspond to consistent process states.

`delta_t_min_k` is a fixed positive design assumption, not an uncertain input or
an economically optimized value. Both utility capacities are nonnegative kW
limits. A changed ΔTmin or capacity creates a different contract. There are no
network control rules in this adapter.

## A combined disturbance crosses the utility limit

```python
model, domain = example()
claim = model.as_claim(domain)
nominal = claim.evaluate_result(domain.scenario("nominal").values)
audit = claim.audit_result()
replayed = claim.evaluate_result(audit.payload.witness.realizations[0])
```

For the supplied synthetic cases, at ΔTmin = 20 K:

| Scenario | Minimum heating, kW | Minimum cooling, kW | Maximum heat recovery, kW |
| --- | ---: | ---: | ---: |
| Nominal | 10 | 40 | 320 |
| Less hot-product flow | 40 | 10 | 290 |
| More cold-feed flow | 40 | 10 | 350 |
| Both changes | 100 | 10 | 290 |

The 50 kW heating limit accommodates either disturbance alone, but the combined
case fails. This is a minimum-utility failure even with unrestricted heat
matching under the declared assumptions. Increasing the limit to 100 kW passes
the target screen for these four cases; it does not demonstrate a workable
exchanger network. Reduced hot-stream flow in this example is not a model of
fouling, which affects exchanger performance and requires additional physics.

The nominal pinch pair is 180 degC on the hot side and 160 degC on the cold side.
With reduced hot-product flow, two pinch pairs appear: 100/80 and 40/20 degC.
The combined case has a 40/20 degC pinch pair. There is no assumption that the
nominal pinch remains the active bottleneck in other scenarios.

## Calculation and exported evidence

The problem-table calculation shifts hot temperatures down by ΔTmin/2 and cold
temperatures up by ΔTmin/2. For each descending temperature interval it
calculates `(sum(hot capacity flows) - sum(cold capacity flows)) * interval width`.
It cascades that heat from an initial value of zero. The largest cumulative
deficit determines the minimum hot utility; adding it makes the cascade
nonnegative. The heat remaining at the bottom is the minimum cold utility.

The scalar output quantities, all in kW, are:

- `minimum_hot_utility` and `minimum_cold_utility`;
- `maximum_heat_recovery`;
- `hot_stream_duty` and `cold_stream_duty`.

The `cascade` evidence contains machine-readable `nodes`, `intervals`,
`pinch_points` and `pinch_intervals`. Pinch points include **all zero-cascade
temperature boundaries**, labelled as interior or terminal. Where the cascade
is zero across a whole interval, `pinch_intervals` records that span as well.
A zero-duty problem has no active temperature nodes or pinch locations.
These records can supply downstream plots without putting plotting in the package.

```python
cascade = next(e for e in nominal.evidence if e.evidence_id == "cascade")
pinches = cascade.details["pinch_points"]
targets = {v.quantity_ref: v.value for v in nominal.payload.values}
```

Calculations use exact rational arithmetic on the declared binary floating-point
values. Evidence retains rational strings as well as display numbers, so very
close temperature nodes remain distinct even if their displayed floats coincide.
Capacity residuals round toward positive infinity before comparison with
`tolerance_kw` (default `1e-8`). This numerical exactness applies to the supplied
model, not to the accuracy of measured stream data.

## Interpret the result scope

The default requirements are `hot_utility_target` and `cold_utility_target`.
Use `requirements=("hot_utility_target",)` to screen only heating capacity.
An evaluation with `feasibility="feasible"` means that the declared heat-cascade
response is defined and its heat balance is satisfied. Inspect the requirement
checks to determine whether its utility targets exceed the selected limits.

The bound contract and audit coverage identify their scope as
`thermodynamic_utility_targets` and record
`installed_network_feasibility="not_assessed"`. A failing audit includes an
`individual_infeasibility` witness scoped to the utility-target requirements,
with membership evidence and a reference to the evaluated cascade. Exceeding a
minimum target rules out meeting that utility limit under the model assumptions;
falling below it is a necessary screen, not sufficient evidence for a particular
network.

Only an explicit `FiniteSet` is supported. `complete_finite` covers precisely
its named scenarios, not the intervals between them or a probability
distribution. An invalid or numerically unresolved scenario prevents a complete
pass. A separately verified failing scenario can still refute the claim, with
partial coverage retained. Evaluation of a nonmember does not run the cascade.

Sensitivity operators, continuous-domain audits, boundary searches and nearest
breaking searches are unsupported. Stream ordering and pinch identity can
change, so derivatives must not be inferred from one active cascade branch.
The adapter also omits variable heat capacity, phase changes, heat losses,
restricted matches, utility-level allocation, heat pumps, storage and dynamics.
Its ideal utilities can supply heating above and cooling below all process
temperatures; actual utility temperature restrictions are not verified.

## Relation to batch and startup studies

Steady-state targeting is useful for comparing throughput, feed-temperature and
production-mix scenarios. Batch heat integration additionally needs the timing
of stream availability. Time slices can represent simultaneous heat exchange;
storage models must account for transfers between times and their temperature
levels. Averaging across time can imply unavailable heat recovery. The current
startup profiles describe scalar loads, without those thermal states, and are
not automatically combined with this adapter.

This implementation follows the established heat-cascade method described in
the [SINTEF/IEA tutorial on process integration](https://pubdb.bfe.admin.ch/en/publication/download/1401).
[Research on heat-exchanger-network operability](https://www.sciencedirect.com/science/article/abs/pii/S0098135413001178)
addresses the additional network flexibility problem.
[Research on batch thermal integration](https://www.sciencedirect.com/science/article/abs/pii/S1359431113006765)
explains time-dependent targeting and storage. These are methodological
references; no third-party implementation was copied or added as a dependency.

Run the consumer example from the checkout:

```bash
uv run python -m examples.pinch
```
