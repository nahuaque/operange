# Model connected equipment and streams

`ProcessNetwork` builds steady-state material and utility balances from named
connections. Its `compile()` method returns an ordinary `LinearProcessAdapter`,
so the same model supports dispatch objectives, continuous audits, equipment
relief, operating-limit searches, controller synthesis and independent
certificate verification. Compilation needs no optional backend and runs no solver.

## Declare the engineering boundary

Every `ProcessStream` has one producer and one consumer. `StreamSource` and
`StreamSink` mark external boundaries. Mixers, headers and splitters conserve
compatible rates; a boiler converts fuel power to steam power at a declared
constant efficiency. Unknown stream rates have finite nonnegative bounds.

The following reconstructs the [two-boiler study](linear-dispatch.md) with
explicit fuel feeds, boiler outlets, a shared header and two delivery branches:

```{literalinclude} ../examples/equipment_network.py
:language: python
:start-at: from operange import (
:end-before: def material_example():
```

The boiler capacities are bounds on `steam_a` and `steam_b`. `StreamTotal("fuel",
...)` adds a reporting sum that can be used by the shared fuel limit and operating
objective. It does not consume either feed a second time.

```python
network, loads = example()
model = network.compile()
claim = model.as_claim(loads)
nominal = claim.evaluate_result(loads.nominal)
audit = claim.audit_result()
```

Nominal minimum fuel is approximately **23 MW**: 12 MW of steam from boiler A
and 4 MW from B. At simultaneous peaks, steam delivery is 20 MW and minimum fuel
is approximately **31 MW**. The 32 MW shared supply passes the continuous box
audit; reducing it to 30 MW produces a verified failure. The example declares a
1e-6 MW objective-gap tolerance, separately from its physical balance tolerances.

## Stream modes and units

| Declaration | Compiled meaning |
| --- | --- |
| `ProcessStream(..., lower=0, upper=12)` | A bounded operating rate, represented by `LinearControl(stream.name, ...)`. A finite upper bound is required. |
| `ProcessStream(..., value=2)` | A fixed rate, represented by equal lower and upper control bounds. Cannot be combined with input or bounds. |
| `ProcessStream(..., input="feed")` | The true uncertainty coordinate directly enters the physical equations. Optional lower/upper bounds become always-applied operating limits; the default lower bound is zero. No adjustable copy of the input is created. |

Rates are extensive quantities: `physical_kind` is `"mass_flow"` or
`"thermal_power"`. The `commodity` label identifies what is conserved, such as
`"water"`, `"steam"` or `"fuel"`. Conserving ports and reporting totals require
identical **unit, physical kind and commodity** labels. Input-bound rates and
sink demand coordinates must use the stream's unit. There is no implicit unit
conversion or inference from labels. Equipment and streams require unique names;
bounded stream names must also be distinct from input names.

These are aggregate rate balances. A water mixer adds water mass flows; it does
not calculate a mixture temperature, pressure or composition. A boiler's steam
stream is thermal power, not steam mass flow. The caller remains responsible
for the chosen energy basis and efficiency assumptions. Property packages,
pressure-flow laws, variable efficiencies and reaction balances are outside this
layer. Storage requires time and inventory equations; the existing
[two-period storage model](storage-replay.md) remains a separate model.

## Equipment equations

| Equipment | Equation or requirement |
| --- | --- |
| `StreamSource(name, outlet)` | External supply boundary; the stream declares its rate or capacity. |
| `Mixer(name, inlets, outlet)` | Outlet rate equals the sum of at least two inlet rates. |
| `UtilityHeader(name, inlets, outlets)` | Total outlet rate equals total inlet rate; branch allocation can adjust. |
| `FixedRatioSplitter(name, inlet, outlets, weights)` | Total outlet rate equals inlet rate; outlet rates are proportional to nonnegative weights. At least two outlets and one positive weight are required. |
| `Boiler(name, fuel, steam, efficiency)` | Steam power equals efficiency times fuel power, with `0 < efficiency <= 1`. Both ports must be thermal power in the same unit. `loss:<name>` reports fuel minus steam. |
| `StreamSink(name, inlet, demand=None, minimum=0, allow_excess=True)` | Delivery must be at least demand input plus minimum. With `allow_excess=False`, an always-applied upper limit also prohibits excess. Omitting demand and minimum gives a passive nonnegative discharge. |

Physical equalities become two opposite `AffineRequirement` operating limits.
The default `balance_tolerance=1e-8` applies to each physical residual in its
stream unit. `balance_scale=1` sets its numerical residual scale, without
changing the tolerance. A sink has its own `tolerance` and `residual_scale`.
Control bounds and input-stream bounds have zero tolerance. Small deviations
within declared physical tolerances are accepted, including a possible small
negative reported boiler loss near unit efficiency.

Splitter weights are **relative ratios**, so `(1, 3)` means a 25/75 split.
The compiler selects the largest positive weight as a pivot and writes
`pivot_weight * branch_rate - branch_weight * pivot_rate = 0` for each other
branch, together with total conservation. It never computes rounded normalized
fractions. A zero weight constrains that branch to zero within the ratio-row
tolerance. Multiplying all weights by a constant preserves the exact ratio but
changes the effective flow tolerance; choose well-scaled dimensionless weights.

```{literalinclude} ../examples/equipment_network.py
:language: python
:start-at: def material_example():
:end-before: def run_example():
```

At 4 kg/s fresh feed, adding 2 kg/s makeup produces 6 kg/s mixed water and
branches of 1.5 and 4.5 kg/s. The fresh feed remains a true physical input;
the fixed makeup cannot be redispatched. Cycles are supported: a splitter branch
can return to a mixer, provided all streams still have one producer, one
consumer and appropriate rate bounds. The solver finds the joint steady state.

## Inspect and extend the compiled model

Generated names remain visible in contracts and evidence:

| Name | Meaning |
| --- | --- |
| `stream:<stream>` | Output alias for a stream's physical rate; also available as `stream.quantity_id`. |
| `balance:<equipment>` | Mixer, header, splitter conservation or boiler conversion residual. |
| `balance:<equipment>:le`, `:ge` | Always-applied bounds on that residual. |
| `ratio:<splitter>:<branch>` | Weighted splitter residual; has corresponding `:le` and `:ge` limits. |
| `delivery:<sink>` | Inlet rate minus the sink's demand input. |
| `service:<sink>` | Selectable delivery requirement; also available as `sink.requirement_id`. |
| `no_excess:<sink>` | Optional always-applied maximum delivery. |
| `stream_bound:<stream>:lower`, `:upper` | Bounds on an input-bound stream. |

Selecting only `requirements=("service:dryer",)` retains every conservation
equation, boiler law, rate bound and no-excess limit. It changes the service
commitment, not the physical network. Unknown, repeated, disconnected and
incompatible ports are rejected before solving.

Use `totals` for named sums of compatible streams. The caller chooses which
streams form a meaningful total; summing both upstream and downstream rates can
double-count delivery. For more specific linear relations, `outputs`,
`requirements` and `operating_limits` accept existing affine declarations.
Custom output terms reference **input coordinates and bounded/fixed stream
names**, rather than other outputs. All generated and caller-supplied names pass
the linear adapter's existing uniqueness and unit checks.

Compilation preserves the exact values of declared float coefficients. If
several input-bound streams share one coordinate, their coefficients are summed
exactly. A combined coefficient that cannot be represented as one finite float
is rejected rather than rounded. Distinct bounded stream variables can express
such relations without folding their coefficients together.

## Use existing queries and controllers

```python
limit = claim.operating_limit("dryer")
search = claim.synthesize_controller()  # requires operange[cvxpy]
if search.verified:
    frozen = search.freeze()
    checked = verify_result(frozen.audit_result())
```

Query capabilities remain those of the compiled adapter and bound domain.
Adjustable audits establish existence of joint steady-state flows. The compiled
controls include **algebraic stream rates**, which need not each correspond to
an independent physical actuator. A saved controller supplies one rule for
every bounded or fixed stream, and its audit checks their consistency with all
balances. This first layer does not eliminate internal flows or infer which
streams are actuated. Input-bound streams need no command rule.

Attach `measurements=MeasurementModel(...)` to restrict controller observations.
Physical stream equations continue to use true inputs; declared measurement
errors cannot enter those equations. As with the underlying adapter, measured
operation requires an explicit or synthesized controller.

Shared relief can target named operating limits such as `shared_fuel`; it cannot
change hard stream bounds. Replacing a stream's `upper` bound and recompiling
expresses a capacity revision for an engineering-change comparison. Conservation
rows also remain ordinary operating limits: explicitly targeting one for relief
changes the physical law and must not be interpreted as an equipment upgrade.
Inspect checked bounds, numerical resolution and the candidate re-audit; a
numerical proposal alone does not establish feasibility or an optimum.

## Save declarations and verify evidence

```python
source_json = network.to_json()
restored = ProcessNetwork.from_json(source_json)
assert restored.compile().base_contract.ref == model.base_contract.ref
report = verify_result(audit.to_json(compact=True))
```

The `process_network/v1` source declaration retains connections, commodity
labels and provenance with a content-derived identity. Loading it executes no
solver or equipment callback. Save it for future editing.

Results and frozen controllers contain the compiled `linear_process/v1` model
and keep their existing formats. Frozen replay and independent verification
need no network compiler. Verification checks the exported physical equations
and evidence; it does not independently certify the translation from the source
network or the suitability of its engineering assumptions. Keep source
declarations alongside the compiled evidence when that correspondence matters.

Run both complete examples, including independent checks of their exports:

```bash
uv run python -m examples.equipment_network
```
